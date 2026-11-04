"""Convert between `aws glue get-workflow`/`get-triggers` and the create/update calls.

The workflow counterpart of prepare_job_update.py, and it exists for the same
reason: the orchestration is where the operational knowledge lives, so leaving
it in the console makes it unreproducible.

Four modes:

  capture  What you run after changing the workflow or a trigger by hand.
           Takes the raw AWS output, strips the read-only fields, keeps only
           the triggers belonging to this workflow, and writes a committable
           file. See infra/glue/capture_jobs.sh.

  check    Validation, safe in CI. Fails on anything that leaks or that AWS
           would reject, warns on anything worth a second look -- including a
           Glue job in infra/glue/ that no trigger in the chain calls.

  names    The trigger names, one per line, so the deploy loop in bash does
           not have to parse JSON.

  emit     One ready-to-send `--cli-input-json` document. The create and
           update shapes genuinely differ: TriggerUpdate cannot carry Type or
           WorkflowName, because Glue will not let a trigger change either, so
           sending the create shape to update-trigger is rejected.

NOTHING HERE IS TEMPLATED, AND THAT IS NOT AN OVERSIGHT
-------------------------------------------------------
A job definition carries the artifact prefix and a role ARN, so
prepare_job_update.py has to template both. A workflow carries neither -- a
trigger references its jobs by bare name. So there is nothing to substitute,
and `check` only has to prove that nothing crept in: it fails on any 12-digit
number, which is what an account id looks like.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys

# `get-workflow` and `get-triggers` return state and identity that the create
# and update calls reject. Stripped on capture rather than on emit, so the
# committed file is already the desired state and a reviewer sees exactly what
# will be sent.
WORKFLOW_READ_ONLY = ("CreatedOn", "LastModifiedOn", "LastRun", "Graph",
                      "BlueprintDetails")
TRIGGER_READ_ONLY = ("Id", "State", "CreatedOn", "LastModifiedOn")

VALID_TYPES = ("SCHEDULED", "CONDITIONAL", "ON_DEMAND")

# Fields `update-trigger` accepts inside TriggerUpdate. Type and WorkflowName
# are deliberately absent: both are immutable once a trigger exists.
TRIGGER_UPDATABLE = ("Description", "Schedule", "Actions", "Predicate",
                     "EventBatchingCondition")

WORKFLOW_FIELDS = ("Name", "Description", "DefaultRunProperties",
                   "MaxConcurrentRuns")


def strip_comments(node):
    """Drop every `_`-prefixed key, at any depth.

    The committed file carries its reasoning inline, the way
    infra/iam/*.json does. AWS rejects unknown keys, so they come out here --
    which also means a comment can never change what is deployed.
    """
    if isinstance(node, dict):
        return {k: strip_comments(v) for k, v in node.items()
                if not k.startswith("_")}
    if isinstance(node, list):
        return [strip_comments(v) for v in node]
    return node


def capture(workflow: dict, triggers: list[dict]) -> dict:
    """Raw AWS output -> a committable definition."""
    name = workflow.get("Name")

    kept = {k: v for k, v in workflow.items()
            if k in WORKFLOW_FIELDS and v is not None}
    dropped = [k for k in WORKFLOW_READ_ONLY if k in workflow]

    mine, foreign = [], 0
    for trigger in triggers:
        if trigger.get("WorkflowName") != name:
            foreign += 1
            continue
        clean = {k: v for k, v in trigger.items()
                 if k not in TRIGGER_READ_ONLY and v is not None}
        # State is read-only, so whether a trigger is armed cannot be captured.
        # Preserve whatever the committed file already said rather than
        # inventing a default -- the caller merges this in.
        mine.append(clean)

    mine.sort(key=lambda t: t.get("Name", ""))

    print(f"workflow {name!r}: dropped {len(dropped)} read-only field(s)"
          f"{': ' + ', '.join(dropped) if dropped else ''}", file=sys.stderr)
    print(f"kept {len(mine)} trigger(s) belonging to it, ignored {foreign} "
          f"belonging to something else", file=sys.stderr)

    return {"Workflow": kept, "Triggers": mine}


def _cron_problems(schedule: str) -> list[str]:
    """Glue takes the six-field EventBridge cron, in UTC."""
    match = re.fullmatch(r"cron\((.+)\)", schedule.strip())
    if not match:
        return [f"schedule {schedule!r} is not of the form cron(...)"]

    fields = match.group(1).split()
    if len(fields) != 6:
        return [f"schedule {schedule!r} has {len(fields)} fields, not 6 "
                f"(minute hour day-of-month month day-of-week year)"]

    day_of_month, day_of_week = fields[2], fields[4]
    if "?" not in (day_of_month, day_of_week):
        return [f"schedule {schedule!r}: one of day-of-month or day-of-week "
                f"must be '?', or AWS rejects it"]
    return []


def check(data: dict, path: str, job_dir: pathlib.Path | None = None) -> int:
    """Validate a committable workflow definition. Returns an exit code.

    Three levels, not two. Errors leak or get rejected; warnings are choices
    worth a second look. Notes are facts a reader needs that no edit can
    clear -- whether the clock is armed is the only one, and it belongs here
    rather than as a warning precisely because it stays true forever: Glue
    reads StartOnCreation at creation and never again, so an armed trigger is
    still described by a file that says false. A warning nobody can action is
    how a reviewer learns to skip the warnings.
    """
    errors, warnings, notes = [], [], []
    clean = strip_comments(data)
    blob = json.dumps(clean)

    leaked = sorted(set(re.findall(r"(?<!\d)\d{12}(?!\d)", blob)))
    if leaked:
        errors.append(f"a 12-digit account id appears in the file: {leaked}")
    if "arn:aws:" in blob:
        errors.append("an ARN appears in the file; a trigger references its "
                      "jobs by bare name and needs none")

    workflow = clean.get("Workflow") or {}
    triggers = clean.get("Triggers") or []
    name = workflow.get("Name")

    if not name:
        errors.append("Workflow.Name is missing")
    if not triggers:
        errors.append("no Triggers -- the workflow would never start")

    seen, called = set(), set()
    scheduled = 0

    for trigger in triggers:
        label = trigger.get("Name") or "<unnamed>"
        if not trigger.get("Name"):
            errors.append("a trigger has no Name")
        elif label in seen:
            errors.append(f"duplicate trigger name {label!r}")
        seen.add(label)

        if trigger.get("WorkflowName") != name:
            errors.append(f"{label}: WorkflowName is "
                          f"{trigger.get('WorkflowName')!r}, not {name!r}")

        kind = trigger.get("Type")
        if kind not in VALID_TYPES:
            errors.append(f"{label}: Type {kind!r} is not one of "
                          f"{list(VALID_TYPES)}")

        actions = trigger.get("Actions") or []
        if not actions:
            errors.append(f"{label}: no Actions -- it would fire and do nothing")
        for action in actions:
            if not action.get("JobName"):
                errors.append(f"{label}: an action has no JobName")
            else:
                called.add(action["JobName"])

        if kind == "SCHEDULED":
            scheduled += 1
            schedule = trigger.get("Schedule")
            if not schedule:
                errors.append(f"{label}: a SCHEDULED trigger needs a Schedule")
            else:
                errors.extend(f"{label}: {p}" for p in _cron_problems(schedule))
            if trigger.get("Predicate"):
                errors.append(f"{label}: a SCHEDULED trigger takes no Predicate")
            armed = "armed" if trigger.get("StartOnCreation") else "NOT armed"
            notes.append(f"{label}: {schedule}, and on creation {armed}. "
                         f"Glue reads StartOnCreation once, so the live state "
                         f"is whatever start-trigger/stop-trigger last set: "
                         f"`aws glue get-trigger --name {label} "
                         f"--query Trigger.State`")

        if kind == "CONDITIONAL":
            predicate = trigger.get("Predicate") or {}
            conditions = predicate.get("Conditions") or []
            if not conditions:
                errors.append(f"{label}: a CONDITIONAL trigger needs a "
                              f"Predicate with Conditions")
            for condition in conditions:
                if not condition.get("JobName"):
                    errors.append(f"{label}: a condition has no JobName")
                if condition.get("State") != "SUCCEEDED":
                    warnings.append(
                        f"{label}: condition state is "
                        f"{condition.get('State')!r}, not 'SUCCEEDED' -- the "
                        f"chain would carry on past a failed upstream job")
            if trigger.get("Schedule"):
                errors.append(f"{label}: a CONDITIONAL trigger takes no Schedule")

    if scheduled == 0:
        warnings.append("no SCHEDULED trigger -- nothing starts this workflow "
                        "except `aws glue start-workflow-run`")
    elif scheduled > 1:
        warnings.append(f"{scheduled} SCHEDULED triggers -- the chain would "
                        f"start more than once a day")

    # The check that actually answers "does it call all three jobs?".
    if job_dir and job_dir.is_dir():
        defined = {p.stem for p in job_dir.glob("*.json")}
        if defined:
            missing = sorted(called - defined)
            if missing:
                errors.append(f"triggers call job(s) with no definition in "
                              f"{job_dir}/: {missing} -- a typo here creates a "
                              f"trigger pointing at a job that does not exist")
            uncalled = sorted(defined - called)
            if uncalled:
                warnings.append(f"job(s) defined but not called by this "
                                f"workflow: {uncalled}")

    label = pathlib.Path(path).name
    for e in errors:
        print(f"  ERROR   {label}: {e}", file=sys.stderr)
    for w in warnings:
        print(f"  warn    {label}: {w}", file=sys.stderr)
    for n in notes:
        print(f"  note    {label}: {n}", file=sys.stderr)
    if not errors and not warnings:
        print(f"  ok      {label}", file=sys.stderr)
    return 1 if errors else 0


def emit_workflow(data: dict) -> dict:
    """The cli-input-json for create-workflow and update-workflow alike.

    Both take the same field set, which is why this needs no --op.
    """
    workflow = strip_comments(data).get("Workflow") or {}
    return {k: v for k, v in workflow.items() if k in WORKFLOW_FIELDS}


def emit_trigger(data: dict, name: str, op: str) -> dict:
    """The cli-input-json for create-trigger or update-trigger."""
    triggers = strip_comments(data).get("Triggers") or []
    found = [t for t in triggers if t.get("Name") == name]
    if not found:
        raise KeyError(
            f"no trigger named {name!r}; this file defines "
            f"{[t.get('Name') for t in triggers]}")
    trigger = found[0]

    if op == "create":
        return trigger

    # update-trigger: the name travels outside the update document, and the
    # document carries only what Glue permits changing.
    update = {k: v for k, v in trigger.items()
              if k in TRIGGER_UPDATABLE and v is not None}
    return {"Name": name, "TriggerUpdate": update}


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mode", choices=["capture", "check", "names", "emit"])
    parser.add_argument("input", help="path to the JSON file")
    parser.add_argument("--triggers",
                        help="capture mode: `aws glue get-triggers` output")
    parser.add_argument("--merge-state",
                        help="capture mode: the committed file, so "
                             "StartOnCreation survives a capture (AWS does "
                             "not return it)")
    parser.add_argument("--job-dir", default="infra/glue",
                        help="check mode: where one-file-per-job definitions "
                             "live, to verify every called job exists")
    parser.add_argument("--kind", choices=["workflow", "trigger"],
                        help="emit mode: which document to produce")
    parser.add_argument("--name", help="emit mode, trigger kind: which trigger")
    parser.add_argument("--op", choices=["create", "update"], default="create",
                        help="emit mode, trigger kind: create and update take "
                             "different shapes")
    args = parser.parse_args()

    data = json.loads(pathlib.Path(args.input).read_text(encoding="utf-8"))

    if args.mode == "check":
        sys.exit(check(data, args.input, pathlib.Path(args.job_dir)))

    if args.mode == "names":
        for trigger in strip_comments(data).get("Triggers") or []:
            print(trigger.get("Name", ""))
        return

    if args.mode == "capture":
        if not args.triggers:
            parser.error("capture mode needs --triggers")
        # get-workflow wraps it; a captured file is already bare.
        workflow = data.get("Workflow", data)
        raw = json.loads(pathlib.Path(args.triggers).read_text(encoding="utf-8"))
        result = capture(workflow, raw.get("Triggers", raw))

        if args.merge_state:
            committed = strip_comments(json.loads(
                pathlib.Path(args.merge_state).read_text(encoding="utf-8")))
            armed = {t.get("Name"): t.get("StartOnCreation")
                     for t in committed.get("Triggers") or []}
            for trigger in result["Triggers"]:
                if trigger["Name"] in armed:
                    trigger["StartOnCreation"] = armed[trigger["Name"]]
            print(f"carried StartOnCreation over for "
                  f"{len(armed)} trigger(s) -- AWS does not return it",
                  file=sys.stderr)
    elif args.kind == "workflow":
        result = emit_workflow(data)
    elif args.kind == "trigger":
        if not args.name:
            parser.error("emit --kind trigger needs --name")
        result = emit_trigger(data, args.name, args.op)
    else:
        parser.error("emit mode needs --kind")

    json.dump(result, sys.stdout, indent=2, default=str)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
