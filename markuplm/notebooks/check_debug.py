check_annotations = {
"The Queen’s Diamond Jubilee Beacons":"https://resource.esriuk.com/esri-resources/the-queens-diamond-jubilee-beacons/",
"KYOCERA SLD Laser, Inc.\nUpdated June 2021.":"https://www.greatplacetowork.com/certified-company/7020473",
"Universal Parks & REsorts": "https://www.gsdm.com/clients/",
"Harry's": "https://www.gsdm.com/harrys-a-man-like-you-case-study/",
"Grupo Martí": "https://www.informatica.com/about-us/customers/customer-success-stories/elkjop.html",
"Lagardère": "https://www.informatica.com/about-us/customers/customer-success-stories/lagardere-travel-retail-pacific.html",
"L'Oréal": "https://www.informatica.com/about-us/customers/customer-success-stories/loreal.html",
"Elkjøp": "https://www.informatica.com/about-us/customers/customer-success-stories/elkjop.html",
"HARNAŚ": "https://www.cortezbrothers.com/michal-sablinski",
}
# #! Found several examples that we lost annotations due to accents and symbols

# %%
from lxml.html.clean import Cleaner
cleaner = Cleaner()
cleaner
# %%
cleaner.clean_html("ASd ")
