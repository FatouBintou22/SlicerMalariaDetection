cmake_minimum_required(VERSION 3.16.3...3.19.7 FATAL_ERROR)

project(SlicerMalariaDetection)

#-----------------------------------------------------------------------------
# Extension meta-information
set(EXTENSION_HOMEPAGE "https://www.slicer.org/wiki/Documentation/Nightly/Extensions/MalariaDetection")
set(EXTENSION_CONTRIBUTORS "Fatou Bintou Ndiaye (Ecole Supérieure Polytechnique)", "Habayatou Diallo (Ecole Supérieure Polytechnique)","Andras Lasso (Queen's University)", "Mouhamed Diop (Ecole Supérieure Polytechnique)", "Mohamed Alalli Bilal (Ecole Supérieure Polytechnique)", "Mamadou Diahame (Ecole Supérieure Polytechnique)", "Mamadou Samba Camara (Ecole Supérieure Polytechnique)", "Sonia Pujol (Brigham and Women’s Hospital, Harvard Medical School)")
set(EXTENSION_DESCRIPTION "Detection and classification of blood cells for malaria diagnosis")
set(EXTENSION_ICONURL "https://www.example.com/Slicer/Extensions/MalariaDetection.png")
set(EXTENSION_SCREENSHOTURLS "https://www.example.com/Slicer/Extensions/MalariaDetection/Screenshots/1.png")
set(EXTENSION_DEPENDS "NA") # Specified as a list or "NA" if no dependencies

#-----------------------------------------------------------------------------
# Extension dependencies
find_package(Slicer REQUIRED)
include(${Slicer_USE_FILE})

#-----------------------------------------------------------------------------
# Extension modules
add_subdirectory(MalariaAutoDetection)
## NEXT_MODULE

#-----------------------------------------------------------------------------
include(${Slicer_EXTENSION_GENERATE_CONFIG})
include(${Slicer_EXTENSION_CPACK})
