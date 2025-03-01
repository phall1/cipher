from pydantic import BaseModel
from enum import Enum


class TopicName(str, Enum):
    EXCLUSIVE = "CB_EXCLUSIVE"
    UKR_UPDATE = "THE_UKRAINE_UPDATE"
    AMERICAS = "AMERICAS"
    EUROPE = "EUROPE"
    MIDDLE_EAST = "MIDDLE_EAST"
    ASIA = "ASIA"
    AFRICA = "AFRICA"
    CYBER_TECH_MARKETS = "CYBER_TECH_MARKETS"


class Paragraph(BaseModel):
    title: str
    content: str
    sources: list[str]  # TODO: will add later


class Topic(BaseModel):
    name: TopicName
    paragraphs: list[Paragraph]


class Report(BaseModel):
    id: str
    topics: list[Topic]
