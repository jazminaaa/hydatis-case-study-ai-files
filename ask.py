"""Ask the assistant one question:  python ask.py "How long is a gift card valid?" """
import sys
from pathlib import Path

from velmora.assistant import Assistant
from velmora.kb import load_kb


def main() -> int:
    if len(sys.argv) < 2 or not sys.argv[1].strip():
        print('usage: python ask.py "your question"')
        return 2
    assistant = Assistant(load_kb(Path(__file__).parent / "ai" / "kb"))
    a = assistant.answer(" ".join(sys.argv[1:]))
    print(a.text)
    print(f"\ncited: {', '.join(a.citations) or '-'}   confidence: {a.confidence:.2f}   top-3: {', '.join(a.ranking[:3])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
