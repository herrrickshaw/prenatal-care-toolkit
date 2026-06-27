"""Problem 2 - Sex-ratio-at-birth (SRB) anomaly analytics.

Foeticide is fought where it is measurable: in the sex ratio at birth. A
naturally occurring population produces roughly 105 male births per 100 female
births (about 48.8% female). Districts, clinics, or time windows that deviate
strongly and *significantly* from that baseline are exactly what PCPNDT
auditors look for.

This module ingests a tabular births dataset and:
  * computes SRB by any grouping dimension (district, clinic, month, ...),
  * runs a binomial significance test against the natural baseline,
  * ranks units by how anomalous (skewed toward males) they are.

It works with public datasets (Census, NFHS, Civil Registration System,
HMIS) once mapped to a simple schema. It never looks at imaging.
"""

from .analyze import (
    SRBConfig,
    compute_srb,
    flag_anomalies,
    from_aggregated_counts,
    load_births,
)
from .adapters import (
    SOURCE_ADAPTERS,
    SOURCE_NOTES,
    adapt,
    from_crs_hmis,
    from_ratio_only,
    from_census_csr,
    from_nfhs,
    from_census_child_06,
)

__all__ = [
    "SRBConfig", "compute_srb", "flag_anomalies", "from_aggregated_counts",
    "load_births", "adapt", "SOURCE_ADAPTERS", "SOURCE_NOTES",
    "from_crs_hmis", "from_ratio_only", "from_census_csr",
    "from_nfhs", "from_census_child_06",
]
