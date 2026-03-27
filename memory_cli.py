import argparse

from memory import MemoryAwareConfig, MemoryAwareInference, SQLiteMemoryStore, build_embedder
from memory.explain import format_trace


def build_parser():
    parser = argparse.ArgumentParser(description="Manage local memories for inference.")
    parser.add_argument("--db-path", default="memory.sqlite")
    parser.add_argument("--user-id", default="default")
    parser.add_argument("--embedder", default="hash-384")

    subparsers = parser.add_subparsers(dest="command", required=True)

    add_parser = subparsers.add_parser("add", help="Add a memory item.")
    add_parser.add_argument("--text", required=True)
    add_parser.add_argument("--type", default="auto")
    add_parser.add_argument("--importance", type=float, default=0.5)
    add_parser.add_argument("--version-group-id", default="")

    list_parser = subparsers.add_parser("list", help="List memories.")
    list_parser.add_argument("--status", default="active")
    list_parser.add_argument("--limit", type=int, default=50)

    search_parser = subparsers.add_parser("search", help="Preview retrieved memories for a query.")
    search_parser.add_argument("--query", required=True)
    search_parser.add_argument("--top-k", type=int, default=12)
    search_parser.add_argument("--max-items", type=int, default=4)
    search_parser.add_argument("--similarity-threshold", type=float, default=0.18)
    search_parser.add_argument("--critic-threshold", type=float, default=0.58)
    search_parser.add_argument("--maybe-threshold", type=float, default=0.48)
    search_parser.add_argument("--token-budget", type=int, default=192)
    search_parser.add_argument("--type-allowlist", default="")
    search_parser.add_argument("--recent-context", default="")
    search_parser.add_argument("--show-prompt", action="store_true")

    delete_parser = subparsers.add_parser("delete", help="Delete a memory item.")
    delete_parser.add_argument("--memory-id", required=True)

    archive_parser = subparsers.add_parser("archive", help="Archive a memory item.")
    archive_parser.add_argument("--memory-id", required=True)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    store = SQLiteMemoryStore(args.db_path)
    try:
        embedder = build_embedder(args.embedder)
        memory = MemoryAwareInference(
            store=store,
            embedder=embedder,
            config=MemoryAwareConfig(
                user_id=args.user_id,
                top_k=getattr(args, "top_k", 12),
                max_items=getattr(args, "max_items", 4),
                similarity_threshold=getattr(args, "similarity_threshold", 0.18),
                critic_threshold=getattr(args, "critic_threshold", 0.58),
                maybe_threshold=getattr(args, "maybe_threshold", 0.48),
                memory_token_budget=getattr(args, "token_budget", 192),
                type_allowlist=getattr(args, "type_allowlist", ""),
            ),
        )

        if args.command == "add":
            record = memory.remember(
                text=args.text,
                memory_type=args.type,
                importance=args.importance,
                version_group_id=args.version_group_id or None,
            )
            print(f"added {record.memory_id} type={record.memory_type} importance={record.importance:.2f}")
            return

        if args.command == "list":
            for record in memory.list_memories(status=args.status, limit=args.limit):
                print(
                    f"{record.memory_id}\t{record.memory_type}\t{record.status}\t"
                    f"importance={record.importance:.2f}\tretrieved={record.times_retrieved}\t{record.text}"
                )
            return

        if args.command == "search":
            prompt, trace, _ = memory.prepare_prompt(
                query_text=args.query,
                recent_context=args.recent_context,
            )
            print(format_trace(trace))
            if args.show_prompt:
                print("\nPrompt preview:\n")
                print(prompt)
            return

        if args.command == "delete":
            memory.forget(args.memory_id)
            print(f"deleted {args.memory_id}")
            return

        if args.command == "archive":
            store.archive_memory(args.memory_id, user_id=args.user_id)
            print(f"archived {args.memory_id}")
            return
    finally:
        store.close()


if __name__ == "__main__":
    main()
