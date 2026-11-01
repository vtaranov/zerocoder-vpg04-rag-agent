"""Консольные команды для трёх пунктов ДЗ VPg04."""
import argparse
import json
import sys
from rag_agent import RAGAgent, cat_fact


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("probe")
    sub.add_parser("cat-fact")
    for command in ("add-url", "search", "ask"):
        sub.add_parser(command).add_argument("text")
    args = parser.parse_args()
    if args.command == "cat-fact":
        result = cat_fact()
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if result["ok"] else 1
    agent = RAGAgent()
    if args.command == "probe":
        result = agent.probe()
    elif args.command == "add-url":
        result = agent.add_url(args.text)
    elif args.command == "search":
        result = agent.search(args.text)
    else:
        result = agent.answer(args.text)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print(f"Ошибка: {type(exc).__name__}. Проверьте настройки, сеть и доступность сервисов.", file=sys.stderr)
        sys.exit(1)
