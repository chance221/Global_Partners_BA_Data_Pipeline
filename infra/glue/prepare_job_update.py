"""Convert between `aws glue get-job` output and `aws glue update-job` input.

Python rather than jq so the same tool runs on Windows and on a CI runner, and
so the field list lives in one reviewable place.

Two directions:

  capture  What you run after creating a job by hand. Strips the fields
           JobUpdate rejects, and replaces the real artifact prefix with
           __ARTIFACT_PREFIX__ so the committed file is not pinned to one
           build. The result is the desired state, ready to commit.

  deploy   What CI runs. Substitutes __ARTIFACT_PREFIX__ for the prefix of the
           commit being deployed. The result is a JobUpdate document.

Why fields are stripped: `get-job` returns read-only metadata that JobUpdate
refuses, and the error names the field rather than the reason, so it reads as a
config problem. MaxCapacity is removed for a different reason -- it is a valid
JobUpdate field, but Glue 2.0+ accepts WorkerType + NumberOfWorkers OR
MaxCapacity, never both, and these jobs use WorkerType.

If update-job still rejects a field, add it to READ_ONLY_FIELDS; the error
message names it.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys

READ_ONLY_FIELDS = (
    "Name",              # passed to update-job as --job-name instead
    "CreatedOn",
    "LastModifiedOn",
    "AllocatedCapacity",  # deprecated alias for MaxCapacity
    "MaxCapacity",        # conflicts with WorkerType + NumberOfWorkers
    "JobRunState",
    "ProfileName",
)

PLACEHOLDER = "__ARTIFACT_PREFIX__"
ACCOUNT_PLACEHOLDER = "__ACCOUNT_ID__"


def _walk_strings(node, fn):
    """Apply fn to every string value in a nested dict/list, in place."""
    if isinstance(node, dict):
        for key, value in node.items():
            if isinstance(value, str):
                node[key] = fn(value)
            else:
                _walk_strings(value, fn)
    elif isinstance(node, list):
        for i, value in enumerate(node):
            if isinstance(value, str):
                node[i] = fn(value)
            else:
                _walk_strings(value, fn)


def capture(job: dict, real_prefix: str, account_id: str | None = None) -> dict:
    """get-job output -> a committable JobUpdate with identifiers templated.

    The account id is templated as well as the artifact prefix. It is not a
    credential -- it appears in every ARN -- but on a public repo there is no
    upside to publishing it, and it lets an attacker enumerate role names by
    the error difference between "does not exist" and "access denied".
    """
    removed = [f for f in READ_ONLY_FIELDS if f in job]
    for field in removed:
        del job[field]

    hits = 0

    def template(value: str) -> str:
        nonlocal hits
        if real_prefix in value:
            hits += 1
            return value.replace(real_prefix, PLACEHOLDER)
        return value

    _walk_strings(job, template)

    if account_id:
        account_hits = 0

        def mask(value: str) -> str:
            nonlocal account_hits
            if account_id in value:
                account_hits += 1
                return value.replace(account_id, ACCOUNT_PLACEHOLDER)
            return value

        _walk_strings(job, mask)
        print(f"masked {account_hits} value(s) containing the account id",
              file=sys.stderr)

    print(f"stripped {len(removed)} read-only field(s): {', '.join(removed) or 'none'}",
          file=sys.stderr)
    print(f"templated {hits} path(s) containing {real_prefix!r}", file=sys.stderr)
    if not hits:
        print("WARNING: prefix not found -- check --real-prefix matches the job",
              file=sys.stderr)
    return job


def deploy(job: dict, prefix: str, account_id: str | None = None) -> dict:
    """Committed JobUpdate -> one pinned to a prefix, with the account filled in.

    Derive the account id rather than storing it:
      aws sts get-caller-identity --query Account --output text
    """
    hits = 0

    def substitute(value: str) -> str:
        nonlocal hits
        if PLACEHOLDER in value:
            hits += 1
            return value.replace(PLACEHOLDER, prefix)
        return value

    _walk_strings(job, substitute)
    print(f"substituted {hits} path(s) -> {prefix}", file=sys.stderr)

    if account_id:
        _walk_strings(job, lambda v: v.replace(ACCOUNT_PLACEHOLDER, account_id))
        print("filled in the account id", file=sys.stderr)
    elif ACCOUNT_PLACEHOLDER in json.dumps(job):
        print(f"WARNING: {ACCOUNT_PLACEHOLDER} left unresolved -- pass --account-id",
              file=sys.stderr)
    if not hits:
        print(f"WARNING: no {PLACEHOLDER} found -- was this file captured?",
              file=sys.stderr)
    return job


def check(job: dict, path: str) -> int:
    """Validate a committable JobUpdate. Returns an exit code.

    Errors are things that leak or that update-job rejects; warnings are cost
    and config choices worth a second look. Safe to run in CI.
    """
    errors, warnings = [], []
    blob = json.dumps(job)

    leaked = sorted(set(re.findall(r"(?<!\d)\d{12}(?!\d)", blob)))
    if leaked:
        errors.append(f"a 12-digit account id appears in the file: {leaked}")

    for token in (PLACEHOLDER, ACCOUNT_PLACEHOLDER):
        if token not in blob:
            warnings.append(f"{token} not found -- was --real-prefix/--account-id passed?")

    present = [f for f in READ_ONLY_FIELDS if f in job]
    if present:
        errors.append(f"update-job rejects these fields: {present}")

    workers = job.get("NumberOfWorkers")
    if workers != 2:
        warnings.append(f"NumberOfWorkers is {workers}, not 2 -- the console "
                        f"defaults to 10, which is 5x the bill here")

    timeout = job.get("Timeout")
    if timeout is None:
        warnings.append("no Timeout -- Glue's default is 2880 minutes (48 hours)")
    elif timeout > 60:
        warnings.append(f"Timeout is {timeout} minutes; this pipeline runs in ~5")

    args = job.get("DefaultArguments", {})
    for pinned in ("--period", "--full"):
        if pinned in args:
            warnings.append(f"{pinned} is pinned in DefaultArguments -- a "
                            f"scheduled run cannot vary it")

    label = pathlib.Path(path).name
    for e in errors:
        print(f"  ERROR   {label}: {e}", file=sys.stderr)
    for w in warnings:
        print(f"  warn    {label}: {w}", file=sys.stderr)
    if not errors and not warnings:
        print(f"  ok      {label}", file=sys.stderr)
    return 1 if errors else 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mode", choices=["capture", "deploy", "check"])
    parser.add_argument("input", help="path to the JSON file")
    parser.add_argument("--real-prefix",
                        help="capture mode: the S3 prefix to replace with the "
                             "placeholder, e.g. s3://bucket/artifacts/manual")
    parser.add_argument("--prefix",
                        help="deploy mode: the S3 prefix to substitute in, "
                             "e.g. s3://bucket/artifacts/<sha>")
    parser.add_argument("--account-id",
                        help="capture: mask this account id. deploy: fill it in. "
                             "Derive with `aws sts get-caller-identity "
                             "--query Account --output text`")
    args = parser.parse_args()

    data = json.loads(pathlib.Path(args.input).read_text(encoding="utf-8"))
    # get-job wraps the job in {"Job": {...}}; a captured file is already bare.
    job = data["Job"] if "Job" in data else data

    if args.mode == "check":
        sys.exit(check(job, args.input))

    if args.mode == "capture":
        if not args.real_prefix:
            parser.error("capture mode needs --real-prefix")
        result = capture(job, args.real_prefix.rstrip("/"), args.account_id)
    else:
        if not args.prefix:
            parser.error("deploy mode needs --prefix")
        result = deploy(job, args.prefix.rstrip("/"), args.account_id)

    json.dump(result, sys.stdout, indent=2, default=str)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
