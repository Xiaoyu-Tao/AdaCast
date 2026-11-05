"""Single JSON result on stdout; diagnostics on stderr."""
import argparse
import contextlib
import json
import sys
import traceback
from pathlib import Path
from .runner import dispatch
from .data import json_safe


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", required=True)
    args = parser.parse_args()
    operation_id = None
    operation = None
    try:
        request = json.loads(Path(args.request).read_text(encoding="utf-8"))
        operation_id = request["operation_id"]
        operation = request.get("operation")
        print(json.dumps({"protocol_version": 1, "operation_id": operation_id, "event": "started"}), file=sys.stderr, flush=True)
        with contextlib.redirect_stdout(sys.stderr):
            result = dispatch(request)
        print(json.dumps({"protocol_version": 1, "operation_id": operation_id, "ok": True, "result": json_safe(result)}, allow_nan=False))
        return 0
    except Exception as error:
        traceback.print_exc(file=sys.stderr)
        print(json.dumps({"protocol_version": 1, "operation_id": operation_id, "ok": False,
                          "error": {"code": "INPUT_INVALID" if operation == "inspect" else "COMPUTATION_FAILED", "message": str(error), "retryable": False}}))
        return 1


if __name__ == "__main__":
    sys.exit(main())
