# Pydantic schema for resume entity extraction

from typing import Optional
from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Structured nested models for rich extraction
# ---------------------------------------------------------------------------

class EducationEntry(BaseModel):
    """One education record (degree + institution pair)."""
    institution: Optional[str] = Field(default=None, description="Name of the educational institution")
    degree: Optional[str] = Field(default=None, description="Degree or qualification obtained")
    field_of_study: Optional[str] = Field(default=None, description="Major or specialization")
    cgpa: Optional[str] = Field(default=None, description="CGPA, GPA, or percentage score")
    location: Optional[str] = Field(default=None, description="City/State/Country of the institution")
    start_year: Optional[str] = Field(default=None, description="Start year or date")
    end_year: Optional[str] = Field(default=None, description="End year or date, or 'Present'")


class ExperienceEntry(BaseModel):
    """One work experience record."""
    company: Optional[str] = Field(default=None, description="Company or organization name")
    role: Optional[str] = Field(default=None, description="Job title or designation")
    location: Optional[str] = Field(default=None, description="Work location")
    start_date: Optional[str] = Field(default=None, description="Start date")
    end_date: Optional[str] = Field(default=None, description="End date or 'Present'")
    description: Optional[str] = Field(default=None, description="Key responsibilities and achievements")
    tech_stack: list[str] = Field(default_factory=list, description="Technologies used")


class ProjectEntry(BaseModel):
    """One project record."""
    name: Optional[str] = Field(default=None, description="Project name")
    tech_stack: list[str] = Field(default_factory=list, description="Technologies used")
    description: Optional[str] = Field(default=None, description="Brief description of the project")
    link: Optional[str] = Field(default=None, description="Project URL or repository link")
    date: Optional[str] = Field(default=None, description="Date or time period")


# ---------------------------------------------------------------------------
# Field specifications (flat fields — backward compatible)
# ---------------------------------------------------------------------------

FIELD_SPECS = {
    "name": {"label": "Name", "kind": "string"},
    "email": {"label": "Email", "kind": "string"},
    "phone": {"label": "Phone", "kind": "string"},
    "location": {"label": "Location", "kind": "string"},
    "linkedin": {"label": "LinkedIn", "kind": "string"},
    "github": {"label": "GitHub", "kind": "string"},
    "portfolio": {"label": "Portfolio/Website", "kind": "string"},
    "twitter": {"label": "Twitter/X", "kind": "string"},
    "other_links": {"label": "Other Links", "kind": "list"},
    "designation": {"label": "Designation", "kind": "list"},
    "companies_worked_at": {"label": "Companies Worked At", "kind": "list"},
    "skills": {"label": "Skills", "kind": "list"},
    "years_of_experience": {"label": "Years Of Experience", "kind": "string"},
    "summary": {"label": "Professional Summary", "kind": "string"},
    "college_name": {"label": "College Name", "kind": "list"},
    "degree": {"label": "Degree", "kind": "list"},
    "graduation_year": {"label": "Graduation Year", "kind": "string"},
    "certifications": {"label": "Certifications", "kind": "list"},
    "projects": {"label": "Projects", "kind": "list"},
    "languages": {"label": "Languages", "kind": "list"},
    "achievements": {"label": "Achievements/Awards", "kind": "list"},
    "publications": {"label": "Publications", "kind": "list"},
    "hobbies": {"label": "Hobbies/Interests", "kind": "list"},
    "references": {"label": "References", "kind": "list"},
    "skills_categorized": {"label": "Skills (By Category)", "kind": "dict"},
    # structured fields
    "education_details": {"label": "Education (Structured)", "kind": "structured"},
    "experience_details": {"label": "Experience (Structured)", "kind": "structured"},
    "projects_detailed": {"label": "Projects (Structured)", "kind": "structured"},
}

STRING_FIELDS = [k for k, v in FIELD_SPECS.items() if v["kind"] == "string"]
LIST_FIELDS = [k for k, v in FIELD_SPECS.items() if v["kind"] == "list"]
STRUCTURED_FIELDS = [k for k, v in FIELD_SPECS.items() if v["kind"] == "structured"]


class ResumeEntity(BaseModel):
    # --- flat string fields ---
    name: Optional[str] = Field(default=None)
    email: Optional[str] = Field(default=None)
    phone: Optional[str] = Field(default=None)
    location: Optional[str] = Field(default=None)
    linkedin: Optional[str] = Field(default=None)
    github: Optional[str] = Field(default=None)
    portfolio: Optional[str] = Field(default=None)
    twitter: Optional[str] = Field(default=None)
    other_links: list[str] = Field(default_factory=list)
    designation: list[str] = Field(default_factory=list)
    companies_worked_at: list[str] = Field(default_factory=list)
    skills: list[str] = Field(default_factory=list)
    years_of_experience: Optional[str] = Field(default=None)
    summary: Optional[str] = Field(default=None)
    college_name: list[str] = Field(default_factory=list)
    degree: list[str] = Field(default_factory=list)
    graduation_year: Optional[str] = Field(default=None)
    certifications: list[str] = Field(default_factory=list)
    projects: list[str] = Field(default_factory=list)
    languages: list[str] = Field(default_factory=list)
    achievements: list[str] = Field(default_factory=list)
    publications: list[str] = Field(default_factory=list)
    hobbies: list[str] = Field(default_factory=list)
    references: list[str] = Field(default_factory=list)
    skills_categorized: dict[str, list[str]] = Field(default_factory=dict)

    # --- structured nested fields ---
    education_details: list[EducationEntry] = Field(default_factory=list)
    experience_details: list[ExperienceEntry] = Field(default_factory=list)
    projects_detailed: list[ProjectEntry] = Field(default_factory=list)

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

    @field_validator("education_details", mode="before")
    @classmethod
    def normalize_education(cls, value):
        if value is None:
            return []
        if isinstance(value, list):
            entries = []
            for item in value:
                if isinstance(item, dict):
                    entries.append(EducationEntry(**item))
                elif isinstance(item, EducationEntry):
                    entries.append(item)
            return entries
        return []

    @field_validator("experience_details", mode="before")
    @classmethod
    def normalize_experience(cls, value):
        if value is None:
            return []
        if isinstance(value, list):
            entries = []
            for item in value:
                if isinstance(item, dict):
                    entries.append(ExperienceEntry(**item))
                elif isinstance(item, ExperienceEntry):
                    entries.append(item)
            return entries
        return []

    @field_validator("projects_detailed", mode="before")
    @classmethod
    def normalize_projects_detailed(cls, value):
        if value is None:
            return []
        if isinstance(value, list):
            entries = []
            for item in value:
                if isinstance(item, dict):
                    entries.append(ProjectEntry(**item))
                elif isinstance(item, ProjectEntry):
                    entries.append(item)
            return entries
        return []

    def backfill_flat_fields(self):
        """Populate flat fields from structured data if they are empty.
        
        This ensures backward compatibility: the LLM fills structured fields,
        and we auto-populate the old flat lists from them.
        """
        # education_details -> college_name, degree, graduation_year
        if self.education_details and not self.college_name:
            self.college_name = [
                e.institution for e in self.education_details
                if e.institution
            ]
        if self.education_details and not self.degree:
            self.degree = [
                e.degree for e in self.education_details
                if e.degree
            ]
        if self.education_details and not self.graduation_year:
            # use the most recent end_year
            years = [e.end_year for e in self.education_details if e.end_year]
            if years:
                self.graduation_year = years[0]

        # experience_details -> companies_worked_at, designation
        if self.experience_details and not self.companies_worked_at:
            self.companies_worked_at = [
                e.company for e in self.experience_details
                if e.company
            ]
        if self.experience_details and not self.designation:
            self.designation = [
                e.role for e in self.experience_details
                if e.role
            ]

        # projects_detailed -> projects
        if self.projects_detailed and not self.projects:
            self.projects = [
                p.name for p in self.projects_detailed
                if p.name
            ]

        return self


def default_selected_fields() -> list[str]:
    return [
        "name", "email", "phone", "location",
        "linkedin", "github", "portfolio", "twitter", "other_links",
        "designation", "companies_worked_at", "skills",
        "years_of_experience", "summary",
        "college_name", "degree", "graduation_year",
        "certifications", "projects", "languages",
        "achievements", "publications", "hobbies", "references",
        "education_details", "experience_details", "projects_detailed",
    ]
