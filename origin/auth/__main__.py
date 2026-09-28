"""Manage accounts (for ORIGIN_AUTH=password).

    python -m origin.auth add-user alice --admin
    python -m origin.auth passwd alice
    python -m origin.auth list
    python -m origin.auth remove alice

Passwords are asked interactively, or read from stdin with --password-stdin.
"""

import argparse
import getpass
import sys

from origin.auth.users import UserError, UserStore
from origin.config import get_settings


def _password(args: argparse.Namespace) -> str:
    if args.password_stdin:
        # PowerShell pipes "text\r\n", with a BOM when its output encoding is UTF-8:
        # either would silently become part of the password.
        return sys.stdin.readline().lstrip("\ufeff").rstrip("\r\n")
    first = getpass.getpass("Password: ")
    if first != getpass.getpass("Repeat it: "):
        raise UserError("the passwords do not match")
    return first


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m origin.auth", description=__doc__.split("\n")[0]
    )
    commands = parser.add_subparsers(dest="command", required=True)
    add = commands.add_parser("add-user", help="create a user")
    add.add_argument("username")
    add.add_argument("--admin", action="store_true", help="may download models (first-run setup)")
    add.add_argument("--password-stdin", action="store_true")
    passwd = commands.add_parser("passwd", help="change a password (signs the user out everywhere)")
    passwd.add_argument("username")
    passwd.add_argument("--password-stdin", action="store_true")
    commands.add_parser("list", help="list users")
    remove = commands.add_parser("remove", help="delete a user (their data stays on disk)")
    remove.add_argument("username")
    args = parser.parse_args(argv)

    users = UserStore(get_settings().sqlite_path)
    try:
        if args.command == "add-user":
            user = users.add(args.username, _password(args), is_admin=args.admin)
            print(f"Created {user.username!r}{' (admin)' if user.is_admin else ''}.")
        elif args.command == "passwd":
            users.set_password(args.username, _password(args))
            print(f"Password of {args.username!r} changed.")
        elif args.command == "list":
            for user in users.all():
                role = "admin" if user.is_admin else "user"
                print(f"{user.id:>4}  {user.username:<24} {role:<6} {user.created_at[:10]}")
        elif args.command == "remove":
            users.remove(args.username)
            print(f"Removed {args.username!r}.")
    except UserError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
