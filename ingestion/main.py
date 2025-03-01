import json

from ingestion.models import Paragraph, Report, Topic, TopicName


def load_report_from_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    return Report(
        id=data["id"],
        topics=[
            Topic(
                name=TopicName(topic["name"]),
                paragraphs=[
                    Paragraph(
                        title=paragraph["title"],
                        content=paragraph["content"],
                        sources=[],  # Placeholder, will be populated later
                    )
                    for paragraph in topic["paragraphs"]
                ],
            )
            for topic in data["topics"]
        ],
    )


def main():
    print(load_report_from_json("./reports/2025-02-27.json"))
    pass


if __name__ == "__main__":
    main()