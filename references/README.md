# Key references

Local copies of the PDFs go in `references/pdf/` (git-ignored: most are copyrighted and must not be redistributed). File names below are the local copies.

## Geometrical acoustics and room acoustics

| Local file | Reference | Used for |
|---|---|---|
| `vorlander2020_auralization.pdf` | M. Vorländer, *Auralization: Fundamentals of Acoustics, Modelling, Simulation, Algorithms and Acoustic Virtual Reality*, 2nd ed., Springer, 2020. | Ray tracing, detectors and hit statistics (§11.3), diffuse rain (Eq. 11.7), hybrid models |
| `kuttruff2016_room_acoustics.pdf` | H. Kuttruff, *Room Acoustics*, 6th ed., CRC Press, 2016. | Wall impedance and Paris's integral, Sabine/Eyring, decay theory |
| `savioja2015_overview_ga.pdf` | L. Savioja, U. P. Svensson, "Overview of geometrical room acoustic modeling techniques," *J. Acoust. Soc. Am.* 138(2), 708–730, 2015. doi:10.1121/1.4926438 | Overview of methods and their limits |
| `schroeder2011_phd_thesis.pdf` | D. Schröder, *Physically Based Real-Time Auralization of Interactive Virtual Environments*, PhD thesis, RWTH Aachen, 2011. | Hybrid ISM/ray tracing as in RAVEN, diffuse rain (Eq. 5.20) |
| `borish1984_image_model.pdf` | J. Borish, "Extension of the image model to arbitrary polyhedra," *J. Acoust. Soc. Am.* 75(6), 1827–1836, 1984. | Image-source construction and visibility tests |
| `dalenback1996_unified_diffuse_specular.pdf` | B.-I. L. Dalenbäck, "Room acoustic prediction based on a unified treatment of diffuse and specular reflection," *J. Acoust. Soc. Am.* 100(2), 899–909, 1996. | Scattering in hybrid models (basis of CATT-Acoustic) |
| `siltanen2012_incoherent_image_sources.pdf` | S. Siltanen, T. Lokki, S. Tervo, L. Savioja, "Modeling incoherent reflections from rough room surfaces with image sources," *J. Acoust. Soc. Am.* 131(6), 2012. | Scattering with image sources |
| `vorlander2000_scattering_coefficients.pdf` | M. Vorländer, E. Mommertz, "Definition and measurement of random-incidence scattering coefficients," *Applied Acoustics* 60(2), 187–199, 2000. doi:10.1016/S0003-682X(99)00056-0 | Definition of the scattering coefficient |
| `thomas2017_wayverb.pdf` | M. R. Thomas, *Wayverb: A Graphical Tool for Hybrid Room Acoustics Simulation*, University of Huddersfield, 2017. | Open-source hybrid implementation, crossover to waveguide |
| `wang2021_dg_room_acoustics.pdf` | H. Wang, *Room acoustic modeling with the time-domain discontinuous Galerkin method*, PhD thesis, TU Eindhoven, 2021. | Wave-based reference below ~500 Hz (future work) |

## Signal processing and decay analysis

| Local file | Reference | Used for |
|---|---|---|
| `schroeder1965_reverberation_time.pdf` | M. R. Schroeder, "New method of measuring reverberation time," *J. Acoust. Soc. Am.* 37(3), 409–412, 1965. doi:10.1121/1.1909343 | Backward integration |

## Geometry

| Local file | Reference | Used for |
|---|---|---|
| `sunday2021_geometry_algorithms.pdf` | D. Sunday, *Practical Geometry Algorithms with C++ Code*, 2021. | Ray–polygon intersection, point-in-polygon |

Not available locally but needed: ISO 3382-1/-2 (decay evaluation), ISO 9613-1 (air absorption), Kulowski 1982 and Lehnert 1993 (ray-tracing errors), Mondet et al. 2020 (absorption to impedance).
