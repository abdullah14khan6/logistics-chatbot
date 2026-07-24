from backend.config.settings import Settings
from backend.knowledge.company_profile import load_company_profile


def test_company_profile_loads_supplied_business_facts() -> None:
    profile = load_company_profile(Settings().company_profile_path)

    assert profile.office_hours.display == "Monday-Saturday, 9:00 AM-6:00 PM"
    assert len(profile.services_offered) == 27
    assert len(profile.value_added_services) == 18
    assert "Amazon FBA" in profile.capabilities.amazon_fba


def test_company_profile_resolves_exact_department_contact() -> None:
    profile = load_company_profile(Settings().company_profile_path)

    resolution = profile.resolve_contact("head of imports")

    assert resolution.exact_match
    assert resolution.contact.name == "Umer Khan"


def test_company_profile_uses_declared_contact_fallback() -> None:
    profile = load_company_profile(Settings().company_profile_path)

    resolution = profile.resolve_contact("sea_freight")

    assert not resolution.exact_match
    assert resolution.contact.id == "head_of_services"
