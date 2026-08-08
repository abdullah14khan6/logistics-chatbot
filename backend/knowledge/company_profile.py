import json
import re
from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field


def normalize_lookup(value: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", " ", value.lower())
    return " ".join(normalized.split())


class OfficeHours(BaseModel):
    model_config = ConfigDict(extra="forbid")

    days: str
    opens: str
    closes: str
    display: str


class CompanyOffice(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    name: str
    city: str
    address: str
    phones: list[str] = Field(default_factory=list)
    hours: str


class CompanyPolicies(BaseModel):
    model_config = ConfigDict(extra="forbid")

    payment_terms: str
    prohibited_items: str
    shipment_delays: str


class CompanyCapabilities(BaseModel):
    model_config = ConfigDict(extra="forbid")

    amazon_fba: str


class CompanyContact(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    name: str
    title: str
    email: str = ""
    phone: str = ""
    aliases: list[str] = Field(default_factory=list)

    def matches(self, requested_role: str) -> bool:
        requested = normalize_lookup(requested_role)
        candidates = [self.id, self.title, *self.aliases]
        return requested in {normalize_lookup(candidate) for candidate in candidates}


class CompanyLeader(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    name: str
    title: str
    department: str
    company: str
    email: str = ""
    aliases: list[str] = Field(default_factory=list)

    def matches(self, query: str) -> bool:
        normalized = normalize_lookup(query)
        candidates = [self.id, self.name, self.title, *self.aliases]
        return any(
            candidate_normalized
            and (
                candidate_normalized == normalized
                or candidate_normalized in normalized
            )
            for candidate in candidates
            if (candidate_normalized := normalize_lookup(candidate))
        )


class SubsidiaryCompany(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    name: str
    relationship: str
    description: str
    aliases: list[str] = Field(default_factory=list)
    relationship_confirmed: bool = False


class ContactResolution(BaseModel):
    contact: CompanyContact
    requested_role: str
    exact_match: bool


class CompanyProfile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    company_name: str
    office_hours: OfficeHours
    offices: list[CompanyOffice] = Field(default_factory=list)
    policies: CompanyPolicies
    capabilities: CompanyCapabilities
    leadership: list[CompanyLeader] = Field(default_factory=list)
    subsidiaries: list[SubsidiaryCompany] = Field(default_factory=list)
    default_contact_role: str
    contacts: list[CompanyContact] = Field(default_factory=list)
    contact_routing: dict[str, str] = Field(default_factory=dict)
    services_offered: list[str] = Field(default_factory=list)
    value_added_services: list[str] = Field(default_factory=list)

    def contact_by_id(self, contact_id: str) -> CompanyContact | None:
        normalized = normalize_lookup(contact_id)
        return next(
            (
                contact
                for contact in self.contacts
                if normalize_lookup(contact.id) == normalized
            ),
            None,
        )

    def resolve_contact(self, requested_role: str) -> ContactResolution:
        requested = requested_role.strip()
        for contact in self.contacts:
            if contact.matches(requested):
                return ContactResolution(
                    contact=contact,
                    requested_role=requested,
                    exact_match=True,
                )

        route_key = normalize_lookup(requested).replace(" ", "_")
        routed_contact_id = self.contact_routing.get(route_key)
        if routed_contact_id:
            routed_contact = self.contact_by_id(routed_contact_id)
            if routed_contact:
                return ContactResolution(
                    contact=routed_contact,
                    requested_role=requested,
                    exact_match=False,
                )

        fallback = self.contact_by_id(self.default_contact_role)
        if fallback is None:
            raise ValueError(
                f"Default contact role {self.default_contact_role!r} is not defined."
            )
        return ContactResolution(
            contact=fallback,
            requested_role=requested,
            exact_match=False,
        )

    def public_emails(self) -> set[str]:
        contact_emails = {
            contact.email.lower() for contact in self.contacts if contact.email
        }
        leadership_emails = {
            leader.email.lower() for leader in self.leadership if leader.email
        }
        return contact_emails | leadership_emails

    def resolve_leader(self, query: str) -> CompanyLeader | None:
        return next((leader for leader in self.leadership if leader.matches(query)), None)


@lru_cache(maxsize=8)
def load_company_profile(path: Path) -> CompanyProfile:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return CompanyProfile.model_validate(raw)
