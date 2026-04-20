#!/usr/bin/env python
# Copyright (c) 2026 GitHub@Apex_rl Developer
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Syntax check script for ApexRL."""

import sys
from pathlib import Path


def check_syntax():
    """Check all Python files for syntax errors."""
    src_dir = Path("src/apexrl")
    errors = []

    if not src_dir.exists():
        print(f"Error: {src_dir} not found")
        return 1

    for py_file in src_dir.rglob("*.py"):
        try:
            with open(py_file, encoding="utf-8") as f:
                compile(f.read(), py_file, "exec")
            print(f"OK: {py_file}")
        except SyntaxError as e:
            errors.append(f"Syntax error in {py_file}: {e}")
            print(f"ERROR: {py_file}: {e}")

    if errors:
        print("\n" + "=" * 50)
        print(f"Found {len(errors)} error(s)")
        for err in errors:
            print(err)
        return 1

    print("\nAll files passed syntax check!")
    return 0


if __name__ == "__main__":
    sys.exit(check_syntax())
