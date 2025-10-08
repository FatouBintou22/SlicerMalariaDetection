import logging
import os
from typing import Optional

import vtk
import qt
import slicer
from slicer.i18n import tr as _
from slicer.i18n import translate
from slicer.ScriptedLoadableModule import *
from slicer.util import VTKObservationMixin
from slicer.parameterNodeWrapper import parameterNodeWrapper
from slicer import vtkMRMLScalarVolumeNode


#
# MalariaAutoDetection
#

class MalariaAutoDetection(ScriptedLoadableModule):
    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = _("Malaria AutoDetection")
        self.parent.categories = [translate("qSlicerAbstractCoreModule", "SlicerMalariaDetection")]
        self.parent.dependencies = []
        self.parent.contributors = ["Fatou Bintou Ndiaye (Ecole Supérieure Polytechnique)", "Habayatou Diallo (Ecole Supérieure Polytechnique)",
                                    "Andras Lasso (Queen's University)", "Mouhamed Diop (Ecole Supérieure Polytechnique)", 
                                    "Mohamed Alalli Bilal (Ecole Supérieure Polytechnique)", "Mamadou Diahame (Ecole Supérieure Polytechnique)", 
                                    "Mamadou Samba Camara (Ecole Supérieure Polytechnique)", "Sonia Pujol (Brigham and Women’s Hospital, Harvard Medical School)"]
        self.parent.helpText = _("""This module detects malaria parasites in microscopic images using a pre-trained YOLOv8 model.""")
        self.parent.acknowledgementText = _("""Developed for automated malaria diagnosis from blood smear images.""")


#
# Parameter Node
#

@parameterNodeWrapper
class MalariaAutoDetectionParameterNode:
    """Module parameters."""
    inputImagePath: str = ""
    modelPath: str = ""


#
# Widget
#

class MalariaAutoDetectionWidget(ScriptedLoadableModuleWidget, VTKObservationMixin):
    def __init__(self, parent=None):
        ScriptedLoadableModuleWidget.__init__(self, parent)
        VTKObservationMixin.__init__(self)
        self.logic = None
        self._parameterNode = None
        self._parameterNodeGuiTag = None
        self.currentInputVolume = None
        self.currentImagePath = None
        self.imageLoaded = False

    def setup(self):
        ScriptedLoadableModuleWidget.setup(self)

        # Load UI
        uiWidget = slicer.util.loadUI(self.resourcePath("UI/MalariaAutoDetection.ui"))
        self.layout.addWidget(uiWidget)
        self.ui = slicer.util.childWidgetVariables(uiWidget)
        uiWidget.setMRMLScene(slicer.mrmlScene)

        # Logic
        self.logic = MalariaAutoDetectionLogic()

        # Connections
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.StartCloseEvent, self.onSceneStartClose)
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.EndCloseEvent, self.onSceneEndClose)
        
        # Connect buttons
        self.ui.loadImageButton.connect("clicked(bool)", self.onLoadImageButton)
        self.ui.analyzeButton.connect("clicked(bool)", self.onAnalyzeButton)
        self.ui.exportButton.connect("clicked(bool)", self.onExportButton)
        
        # Connect model selection
        self.ui.modelSelection.connect("currentIndexChanged(int)", self.onModelSelectionChanged)

        # Initialize parameter node
        self.initializeParameterNode()

        # Initialize UI state
        self.ui.outputsCollapsibleButton.collapsed = True
        self.ui.advancedCollapsibleButton.collapsed = True
        self.ui.analyzeButton.enabled = False
        self.ui.exportButton.enabled = False
        
        # Set initial model selection to placeholder
        self.ui.modelSelection.setCurrentIndex(0)
        
        # Disable the placeholder item in combobox
        model_item = self.ui.modelSelection.model().item(0)
        if model_item:
            model_item.setFlags(model_item.flags() & ~qt.Qt.ItemIsEnabled)
        
        # Set initial image status with grey badge
        self.setImageStatus(loaded=False)

    def cleanup(self):
        self.removeObservers()

    def onSceneStartClose(self, caller, event):
        self.setParameterNode(None)

    def onSceneEndClose(self, caller, event):
        if self.parent.isEntered:
            self.initializeParameterNode()

    def initializeParameterNode(self):
        self.setParameterNode(self.logic.getParameterNode())

    def setParameterNode(self, inputParameterNode: Optional[MalariaAutoDetectionParameterNode]):
        if self._parameterNode:
            self._parameterNode.disconnectGui(self._parameterNodeGuiTag)
        self._parameterNode = inputParameterNode
        if self._parameterNode:
            self._parameterNodeGuiTag = self._parameterNode.connectGui(self.ui)

    def setImageStatus(self, loaded=False, filename=""):
        """Update image status badge with grey (not loaded) or green (loaded)"""
        if loaded and filename:
            self.ui.imagePathLabel.setText(f"✅ {filename}")
            self.ui.imagePathLabel.setStyleSheet("""
                QLabel { 
                    color: #2E7D32; 
                    font-size: 12px;
                    font-weight: 500;
                    padding: 8px 12px;
                    background-color: #E8F5E9;
                    border-radius: 4px;
                    margin-top: 5px;
                    border: 1px solid #C8E6C9;
                }
            """)
            self.imageLoaded = True
        else:
            self.ui.imagePathLabel.setText("❌ No files selected")
            self.ui.imagePathLabel.setStyleSheet("""
                QLabel { 
                    color: #757575; 
                    font-size: 12px;
                    font-weight: 500;
                    padding: 8px 12px;
                    background-color: #F5F5F5;
                    border-radius: 4px;
                    margin-top: 5px;
                    border: 1px solid #E0E0E0;
                }
            """)
            self.imageLoaded = False
        
        # Update analyze button state
        self.updateAnalyzeButtonState()

    def onModelSelectionChanged(self, index):
        """Handle model selection change"""
        # Update analyze button state
        self.updateAnalyzeButtonState()
        
        if index > 0:  # Valid model selected (not placeholder)
            model_name = self.ui.modelSelection.currentText
            logging.info(f"Model selected: {model_name}")

    def updateAnalyzeButtonState(self):
        """Enable analyze button only if image is loaded AND valid model is selected"""
        model_selected = self.ui.modelSelection.currentIndex > 0
        can_analyze = self.imageLoaded and model_selected
        self.ui.analyzeButton.enabled = can_analyze
        
        if can_analyze:
            logging.info("Ready to analyze: Image loaded and model selected")

    def onLoadImageButton(self):
        """Load microscopic image from file (PNG, JPG, TIFF, etc.)"""
        # Open file dialog for image selection
        fileDialog = qt.QFileDialog()
        fileDialog.setFileMode(qt.QFileDialog.ExistingFile)
        fileDialog.setNameFilter("Images (*.png *.jpg *.jpeg *.tiff *.tif *.bmp)")
        fileDialog.setWindowTitle("Sélectionner une image microscopique")
        
        if fileDialog.exec_():
            filePaths = fileDialog.selectedFiles()
            if filePaths:
                imagePath = filePaths[0]
                self._loadImageFromPath(imagePath)

    def _loadImageFromPath(self, imagePath):
        """Load 2D microscopic image from file path into Slicer"""
        try:
            logging.info(f"Loading image from: {imagePath}")
            
            # Remove previous volume if exists
            if self.currentInputVolume:
                slicer.mrmlScene.RemoveNode(self.currentInputVolume)
                self.currentInputVolume = None
            
            # Load image using PIL or OpenCV for 2D microscopic images
            import numpy as np
            image_array = None
            
            try:
                # Try PIL first
                from PIL import Image
                img = Image.open(imagePath)
                image_array = np.array(img)
                logging.info(f"Image loaded with PIL. Shape: {image_array.shape}, dtype: {image_array.dtype}")
            except ImportError:
                # Fallback to OpenCV
                import cv2
                image_array = cv2.imread(imagePath)
                if image_array is not None:
                    image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
                    logging.info(f"Image loaded with OpenCV. Shape: {image_array.shape}, dtype: {image_array.dtype}")
            
            if image_array is None:
                raise Exception("Impossible de lire l'image. Vérifiez le format du fichier.")
            
            # Ensure uint8 format
            if image_array.dtype != np.uint8:
                image_array = (image_array * 255).astype(np.uint8) if image_array.max() <= 1.0 else image_array.astype(np.uint8)
            
            # Create volume node from numpy array
            self.currentInputVolume = self.logic.createVolumeFromArray(
                image_array, 
                os.path.splitext(os.path.basename(imagePath))[0]
            )
            
            if self.currentInputVolume:
                # Store the image path
                self.currentImagePath = imagePath
                
                # Update UI with green badge
                fileName = os.path.basename(imagePath)
                self.setImageStatus(loaded=True, filename=fileName)
                
                # Store path in parameter node
                self._parameterNode.inputImagePath = imagePath
                
                # Display image in slice viewers (Red, Yellow, Green)
                slicer.util.setSliceViewerLayers(background=self.currentInputVolume)
                
                # Configure slice views for 2D display
                layoutManager = slicer.app.layoutManager()
                for viewName in ["Red", "Yellow", "Green"]:
                    sliceWidget = layoutManager.sliceWidget(viewName)
                    if sliceWidget:
                        sliceLogic = sliceWidget.sliceLogic()
                        sliceNode = sliceLogic.GetSliceNode()
                        
                        # Set orientation to show XY plane (axial view)
                        sliceNode.SetOrientationToAxial()
                        
                        # Fit slice to all
                        sliceLogic.FitSliceToAll()
                        
                        # Reset slice offset to center
                        sliceLogic.SetSliceOffset(0)
                
                # Reset field of view
                slicer.util.resetSliceViews()
                
                # Déterminer le message selon l'état du modèle
                message_selection = (
                    "Sélectionnez un modèle pour continuer." 
                    if self.ui.modelSelection.currentIndex == 0 
                    else "Vous pouvez maintenant lancer l'analyse."
                )
                
                logging.info(f"Image loaded successfully: {fileName}")
                slicer.util.infoDisplay(
                    f"✓ Image chargée avec succès!\n\n"
                    f"Fichier: {fileName}\n\n"
                    f"{message_selection}",
                    windowTitle="Image chargée"
                )
            else:
                raise Exception("Impossible de créer le volume à partir de l'image")
                
        except Exception as e:
            logging.error(f"Error loading image: {str(e)}")
            slicer.util.errorDisplay(
                f"Erreur lors du chargement de l'image:\n\n{str(e)}\n\n"
                f"Vérifiez que le fichier est une image valide.",
                windowTitle="Erreur de chargement"
            )
            self.setImageStatus(loaded=False)
            self.currentImagePath = None

    def onAnalyzeButton(self):
        """Execute malaria detection when analyze button is clicked"""
        if not self.currentInputVolume:
            slicer.util.errorDisplay(
                "Aucune image chargée.\n\n"
                "Veuillez charger une image microscopique d'abord.",
                windowTitle="Image manquante"
            )
            return
        
        if self.ui.modelSelection.currentIndex == 0:
            slicer.util.errorDisplay(
                "Aucun modèle sélectionné.\n\n"
                "Veuillez sélectionner un modèle de détection.",
                windowTitle="Modèle manquant"
            )
            return
        
        # Get model path
        model_path = self.logic.getDefaultModelPath()
        if not os.path.exists(model_path):
            slicer.util.errorDisplay(
                f"⚠ Fichier de modèle non trouvé!\n\n"
                f"Emplacement attendu:\n{model_path}\n\n"
                f"Veuillez placer votre modèle YOLOv8 (.pt ou .pth) entraîné pour la détection de malaria à cet emplacement.",
                windowTitle="Modèle manquant"
            )
            return
        
        try:
            with slicer.util.tryWithErrorDisplay(_("Échec de l'analyse."), waitCursor=True):
                # Run inference
                logging.info("Starting inference...")
                results = self.logic.runInference(self.currentInputVolume, model_path)
                
                # Display results
                self.displayResults(results)
                
                # Show outputs section
                self.ui.outputsCollapsibleButton.collapsed = False
                
                # Enable export button
                self.ui.exportButton.enabled = True
                
                # Success message
                slicer.util.infoDisplay(
                    f"✓ Analyse terminée avec succès!\n\n"
                    f"🦠 Parasites détectés: {results['parasite_count']}\n"
                    f"⚪ Leucocytes détectés: {results['leukocyte_count']}\n"
                    f"📊 Densité parasitaire: {results['parasitic_density']}\n\n"
                    f"⏱️ Temps de traitement: {results['processing_time']:.2f}s\n\n"
                    f"Les résultats avec annotations sont affichés dans les vues Rouge, Jaune et Verte.",
                    windowTitle="Analyse terminée"
                )
                
        except Exception as e:
            logging.error(f"Analysis failed: {str(e)}")
            slicer.util.errorDisplay(
                f"Erreur pendant l'analyse:\n\n{str(e)}",
                windowTitle="Erreur d'analyse"
            )

    def displayResults(self, results):
        """Display detection results in the UI"""
        # Update result labels
        self.ui.parasiteCountLabel.text = str(results['parasite_count'])
        self.ui.leukocyteCountLabel.text = str(results['leukocyte_count'])
        self.ui.parasiticDensityLabel.text = results['parasitic_density']
        
        # Display annotated image in slice viewers
        if results['annotated_volume']:
            annotatedVolume = results['annotated_volume']
            
            # Show annotated image in all three slice viewers
            slicer.util.setSliceViewerLayers(
                background=annotatedVolume,
                foreground=None,
                label=None
            )
            
            # Configure slice views to display results properly
            layoutManager = slicer.app.layoutManager()
            for viewName in ["Red", "Yellow", "Green"]:
                sliceWidget = layoutManager.sliceWidget(viewName)
                if sliceWidget:
                    sliceLogic = sliceWidget.sliceLogic()
                    sliceNode = sliceLogic.GetSliceNode()
                    
                    # Set to axial orientation
                    sliceNode.SetOrientationToAxial()
                    
                    # Fit to window
                    sliceLogic.FitSliceToAll()
                    
                    # Center the slice
                    sliceLogic.SetSliceOffset(0)
            
            # Reset views
            slicer.util.resetSliceViews()
            
            logging.info("Results displayed in slice viewers")

    def onExportButton(self):
        """Export results to CSV file"""
        if not self.logic.lastResults:
            slicer.util.warningDisplay(
                "Aucun résultat à exporter.\n\n"
                "Veuillez d'abord lancer une analyse.",
                windowTitle="Pas de résultats"
            )
            return
            
        try:
            # Get save location with default filename
            defaultFileName = "malaria_detection_results.csv"
            if self.currentImagePath:
                baseName = os.path.splitext(os.path.basename(self.currentImagePath))[0]
                defaultFileName = f"{baseName}_results.csv"
            
            filename = qt.QFileDialog.getSaveFileName(
                self.parent, 
                "Exporter les résultats", 
                defaultFileName, 
                "CSV Files (*.csv)"
            )
            
            if filename:
                self.logic.exportResults(filename, self.currentImagePath)
                slicer.util.infoDisplay(
                    f"✓ Résultats exportés avec succès!\n\n"
                    f"Fichier: {os.path.basename(filename)}\n"
                    f"Emplacement: {os.path.dirname(filename)}",
                    windowTitle="Export réussi"
                )
                logging.info(f"Results exported to: {filename}")
                
        except Exception as e:
            logging.error(f"Export failed: {str(e)}")
            slicer.util.errorDisplay(
                f"Erreur lors de l'export:\n\n{str(e)}",
                windowTitle="Erreur d'export"
            )


#
# Logic
#

class MalariaAutoDetectionLogic(ScriptedLoadableModuleLogic):
    def __init__(self):
        ScriptedLoadableModuleLogic.__init__(self)
        self.lastResults = None

    def getParameterNode(self):
        return MalariaAutoDetectionParameterNode(super().getParameterNode())

    
    def getDefaultModelPath(self):
        # Get module folder dynamically
        moduleDir = os.path.dirname(slicer.util.modulePath('MalariaAutoDetection'))
        # Construct model path
        modelPath = os.path.join(moduleDir, 'Resources', 'Models', 'malariaDetection.pth')
        # Create folder if it doesn't exist
        os.makedirs(os.path.dirname(modelPath), exist_ok=True)
        return modelPath


    def createVolumeFromArray(self, image_array, name="MicroscopicImage"):
        """Create a MRML volume node from numpy array (for loading images)"""
        import numpy as np
        from vtk.util import numpy_support
        import vtk
        import slicer

        # Ensure RGB image
        if len(image_array.shape) == 2:
            import cv2
            image_array = cv2.cvtColor(image_array, cv2.COLOR_GRAY2RGB)
        elif image_array.shape[2] == 4:  # RGBA
            import cv2
            image_array = cv2.cvtColor(image_array, cv2.COLOR_RGBA2RGB)

        # Create new volume node - UTILISER vtkMRMLVectorVolumeNode pour RGB
        volumeNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLVectorVolumeNode")
        volumeNode.SetName(name)

        # Convert numpy array to VTK
        vtk_array = numpy_support.numpy_to_vtk(
            image_array.ravel(),
            deep=True,
            array_type=vtk.VTK_UNSIGNED_CHAR
        )
        vtk_array.SetNumberOfComponents(3)

        # Create image data
        imageData = vtk.vtkImageData()
        imageData.SetDimensions(image_array.shape[1], image_array.shape[0], 1)
        imageData.GetPointData().SetScalars(vtk_array)

        # Set to volume node
        volumeNode.SetAndObserveImageData(imageData)
        volumeNode.CreateDefaultDisplayNodes()

        # Set proper spacing for 2D microscopic images
        volumeNode.SetSpacing(1.0, 1.0, 1.0)
        volumeNode.SetOrigin(0.0, 0.0, 0.0)

        return volumeNode
    

    def runInference(self, inputVolume, modelPath):
        """
        Run YOLOv8 inference on the input image
        """
        
        if not inputVolume:
            raise ValueError("Input volume is invalid")
        
        import time
        import numpy as np
        import logging
        from vtk.util import numpy_support
        import cv2
        from ultralytics import YOLO
        
        startTime = time.time()
        logging.info("Starting malaria detection inference")
    
        if not os.path.exists(modelPath):
            raise FileNotFoundError(f"Model file not found: {modelPath}")
        
        try:
            # Load YOLO model only once per session (performance)
            if not hasattr(self, "_yoloModel") or self._yoloModel.model.fuse is None:
                self._yoloModel = YOLO(modelPath)
                logging.info(f"YOLO model loaded from: {modelPath}")

            model = self._yoloModel

            # Convert inputVolume (VTK) → numpy
            imageData = inputVolume.GetImageData()
            dims = imageData.GetDimensions()
            scalars = imageData.GetPointData().GetScalars()

            image_array = numpy_support.vtk_to_numpy(scalars)
            numComponents = scalars.GetNumberOfComponents()

            if numComponents == 1:
                image_array = image_array.reshape(dims[1], dims[0])
                image_array = cv2.cvtColor(image_array.astype(np.uint8), cv2.COLOR_GRAY2RGB)
            elif numComponents == 3:
                image_array = image_array.reshape(dims[1], dims[0], 3)
            elif numComponents == 4:
                image_array = image_array.reshape(dims[1], dims[0], 4)
                image_array = cv2.cvtColor(image_array.astype(np.uint8), cv2.COLOR_RGBA2RGB)

            image_array = image_array.astype(np.uint8)

            # Run YOLO inference
            results = model.predict(source=image_array, conf=0.25, verbose=False)
            res = results[0]

            # Count detections
            parasite_count = 0
            leukocyte_count = 0
            if len(res.boxes) > 0:
                for box in res.boxes:
                    cls_id = int(box.cls[0])
                    if cls_id == 0:
                        parasite_count += 1
                    elif cls_id == 1:
                        leukocyte_count += 1

            # Parasitic density
            if leukocyte_count > 0:
                density = (parasite_count / leukocyte_count) * 8000
                density_str = f"{density:.0f} parasites/µL"
            else:
                density_str = "N/A (aucun leucocyte détecté)"

            # Create annotated image
            annotated_image = res.plot()
            annotated_volume = self.createVolumeFromArray(
                annotated_image,
                "MalariaDetection_Result"
            )

            # Store results
            self.lastResults = {
                'parasite_count': parasite_count,
                'leukocyte_count': leukocyte_count,
                'parasitic_density': density_str,
                'annotated_volume': annotated_volume,
                'raw_results': res,
                'processing_time': time.time() - startTime
            }

            logging.info(f"✅ Inference complete in {self.lastResults['processing_time']:.2f}s")
            logging.info(f"Found {parasite_count} parasites and {leukocyte_count} leukocytes")
            
            return self.lastResults

        except Exception as e:         
            logging.error(f"Error during inference: {str(e)}")
            raise RuntimeError(f"Error during YOLO inference: {str(e)}")

    def exportResults(self, filename, imagePath=None):
        """Export detection results to CSV file"""
        if not self.lastResults:
            raise ValueError("No results to export")
        
        import csv
        with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Métrique', 'Valeur'])
            if imagePath:
                writer.writerow(['Fichier image', os.path.basename(imagePath)])
            writer.writerow(['Nombre de parasites', self.lastResults['parasite_count']])
            writer.writerow(['Nombre de leucocytes', self.lastResults['leukocyte_count']])
            writer.writerow(['Densité parasitaire', self.lastResults['parasitic_density']])
            writer.writerow(['Temps de traitement (secondes)', f"{self.lastResults['processing_time']:.2f}"])


#
# Tests
#

class MalariaAutoDetectionTest(ScriptedLoadableModuleTest):
    def setUp(self):
        slicer.mrmlScene.Clear()

    def runTest(self):
        self.setUp()
        self.test_MalariaAutoDetection1()

    def test_MalariaAutoDetection1(self):
        self.delayDisplay("Starting test")
        logic = MalariaAutoDetectionLogic()

        # Test with a sample image
        try:
            import numpy as np
            # Create a dummy test image
            test_image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
            testVolume = logic.createVolumeFromArray(test_image, "TestImage")

            self.assertIsNotNone(testVolume)
            self.delayDisplay("Volume creation test passed")

            # Check model path
            modelPath = logic.getDefaultModelPath()
            if os.path.exists(modelPath):
                results = logic.runInference(testVolume, modelPath)
                self.assertIsNotNone(results)
                self.delayDisplay("Inference test passed")
            else:
                self.delayDisplay("Test skipped: Model file not found")

        except Exception as e:
            self.delayDisplay(f"Test failed: {str(e)}")