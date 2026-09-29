"""`python -m origin`: the same as the `origin` command, for when a generated launcher
(origin.exe) is blocked, as Windows Smart App Control does with unsigned executables."""

from origin.app import run

run()
