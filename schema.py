"""Pydantic schema used by the app and parser."""

from typing import Optional

from pydantic import BaseModel, Field, field_validator

FIELD_SPECS = {
    "name": {"label": "Name", "kind": "string"},
    "email": {"label": "Email", "kind": "string"},
    "phone": {"label": "Phone", "kind": "string"},
    "location": {"label": "Location", "kind": "string"},
    "linkedin": {"label": "LinkedIn", "kind": "string"},
    "github": {"label": "GitHub", "kind": "string"},
    "portfolio": {"label": "Portfolio", "kind": "string"},
    "designation": {"label": "Designation", "kind": "list"},
    "companies_worked_at": {"label": "Companies Worked At", "kind": "list"},
    "skills": {"label": "Skills", "kind": "list"},
    "college_name": {"label": "College Name", "kind": "list"},
    "degree": {"label": "Degree", "kind": "list"},
    "graduation_year": {"label": "Graduation Year", "kind": "string"},
    "years_of_experience": {"label": "Years Of Experience", "kind": "string"},
    "certifications": {"label": "Certifications", "kind": "list"},
    "projects": {"label": "Projects", "kind": "list"},
    "languages": {"label": "Languages", "kind": "list"},
    "summary": {"label": "Professional Summary", "kind": "string"},
}

STRING_FIELDS = [k for k, v in FIELD_SPECS.items() if v["kind"] == "string"]
LIST_FIELDS = [k for k, v in FIELD_SPECS.items() if v["kind"] == "list"]


class ResumeEntity(BaseModel):
    name: Optional[str] = Field(default=None)
    email: Optional[str] = Field(default=None)
    phone: Optional[str] = Field(default=None)
    location: Optional[str] = Field(default=None)
    linkedin: Optional[str] = Field(default=None)
    github: Optional[str] = Field(default=None)
    portfolio: Optional[str] = Field(default=None)
    designation: list[str] = Field(default_factory=list)
    companies_worked_at: list[str] = Field(default_factory=list)
    skills: list[str] = Field(default_factory=list)
    college_name: list[str] = Field(default_factory=list)
    degree: list[str] = Field(default_factory=list)
    graduation_year: Optional[str] = Field(default=None)
    years_of_experience: Optional[str] = Field(default=None)
    certifications: list[str] = Field(default_factory=list)
    projects: list[str] = Field(default_factory=list)
    languages: list[str] = Field(default_factory=list)
    summary: Optional[str] = Field(default=None)

    @field_validator(*LIST_FIELDS, mode="before")
    @classmethod
    def normalize_list_fields(cls, value):
        if value is None:
            return []
        if isinstance(value, str):
            parts = [item.strip() for item in value.split(",")]
            return [item for item in parts if item]
        if isinstance(value, list):
            cleaned = []
            for item in value:
                if item is None:
                    continue
                text = str(item).strip()
                if text:
                    cleaned.append(text)
            return cleaned
        return []


def default_selected_fields() -> list[str]:
    """Default list shown in UI for extraction."""
    return [
        "name",
        "email",
        "phone",
        "location",
        "designation",
        "companies_worked_at",
        "skills",
        "college_name",
        "degree",
        "graduation_year",
        "years_of_experience",
        "certifications",
        "projects",
        "languages",
        "linkedin",
        "github",
    ]
