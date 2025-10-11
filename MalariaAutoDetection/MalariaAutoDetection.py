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

    def setup(self):
        ScriptedLoadableModuleWidget.setup(self)

        # Load UI
        uiWidget = slicer.util.loadUI(self.resourcePath("UI/MalariaAutoDetection.ui"))
        self.layout.addWidget(uiWidget)
        self.ui = slicer.util.childWidgetVariables(uiWidget)
        uiWidget.setMRMLScene(slicer.mrmlScene)

        # Logic
        self.logic = MalariaAutoDetectionLogic()

        models = self.logic.getModels()
        for model_name, model_path in models:
            self.ui.modelSelection.addItem(model_name, model_path)

        # Connections
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.StartCloseEvent, self.onSceneStartClose)
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.EndCloseEvent, self.onSceneEndClose)
        
        # Connect buttons
        self.ui.analyzeButton.connect("clicked(bool)", self.onAnalyzeButton)
        self.ui.exportButton.connect("clicked(bool)", self.onExportButton)
        self.ui.inputImageNodeSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.updateAnalyzeButtonState)
        self.ui.modelSelection.connect("currentIndexChanged(int)", self.updateAnalyzeButtonState)

        # Initialize parameter node
        self.initializeParameterNode()

        # Initialize UI state
        self.ui.outputsCollapsibleButton.collapsed = True
        self.ui.advancedCollapsibleButton.collapsed = True

        self.ui.exportButton.enabled = False

        self.updateAnalyzeButtonState()

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

    def updateAnalyzeButtonState(self):
        """Enable analyze button only if image is loaded AND valid model is selected"""
        can_analyze = self.currentInputVolume and (self.ui.modelSelection.currentIndex >= 0)
        self.ui.analyzeButton.enabled = can_analyze
        if can_analyze:
            logging.info("Ready to analyze: Image loaded and model selected")

    def onAnalyzeButton(self):
        """Execute malaria detection when analyze button is clicked"""
        
        with slicer.util.tryWithErrorDisplay(_("Échec de l'analyse."), waitCursor=True):

            if not self.currentInputVolume:
                raise RuntimeError("Aucune image chargée. Veuillez charger une image microscopique d'abord.")
            
            # Get model path
            model_path = self.ui.modelSelection.currentData
            if not os.path.exists(model_path):
                raise RuntimeError(f"Fichier de modèle non trouvé! Emplacement attendu: {model_path}")

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
            f"⏱️ Temps de traitement: {results['processing_time']:.2f}s\n",
            windowTitle="Analyse terminée"
        )

    def displayResults(self, results):
        """Display detection results in the UI"""
        # Update result labels
        self.ui.parasiteCountLabel.text = str(results['parasite_count'])
        self.ui.leukocyteCountLabel.text = str(results['leukocyte_count'])
        self.ui.parasiticDensityLabel.text = results['parasitic_density']
        
        slicer.app.layoutManager().setLayout(slicer.vtkMRMLLayoutNode.SlicerLayoutOneUpRedSliceView)
        slicer.util.setSliceViewerLayers(background=self.currentInputVolume, fit=True)

    @property
    def currentInputVolume(self):
        return self.ui.inputImageNodeSelector.currentNode()

    @property
    def currentImagePath(self):
        volumeNode = self.ui.inputImageNodeSelector.currentNode()
        if volumeNode and volumeNode.GetStorageNode():
            return volumeNode.GetStorageNode().GetFileName()
        return ""

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

    RESOURCES_PATH = os.path.join(os.path.dirname(__file__),  "Resources")

    def __init__(self):
        ScriptedLoadableModuleLogic.__init__(self)
        self.lastResults = None

    def getParameterNode(self):
        return MalariaAutoDetectionParameterNode(super().getParameterNode())

    def getModels(self):
        """Return list of model name and path pairs"""
        models = []
        models_dir = os.path.join(self.RESOURCES_PATH, "Models")

        # If the models directory doesn't exist, return empty list
        if not os.path.isdir(models_dir):
            raise FileNotFoundError(f"Models directory not found: {models_dir}")

        try:
            for fname in os.listdir(models_dir):
                if not fname:
                    continue
                lower = fname.lower()
                #if lower.endswith('.pt') or lower.endswith('.pth'):
                if lower.endswith('.pt'):
                    full_path = os.path.abspath(os.path.join(models_dir, fname))
                    name = os.path.splitext(fname)[0]
                    models.append((name, full_path))
        except Exception:
            # On any error (permissions, etc.) return what we have so far
            return models

        # Sort by model name for predictable ordering
        models.sort(key=lambda x: x[0].lower())
        return models

    def getDefaultModelPath(self):
        # Get module folder dynamically
        moduleDir = os.path.dirname(slicer.util.modulePath('MalariaAutoDetection'))
        # Construct model path
        modelPath = os.path.join(moduleDir, 'Resources', 'Models', 'malariaDetection.pth')
        # Create folder if it doesn't exist
        os.makedirs(os.path.dirname(modelPath), exist_ok=True)
        return modelPath

    def runInference(self, inputVolume, modelPath):
        """
        Run YOLOv8 inference on the input image
        """
        
        if not inputVolume:
            raise ValueError("Input volume is invalid")

        import time
        import numpy as np
        import logging

        try:
            from ultralytics import YOLO
        except ImportError:
            slicer.util.pip_install('ultralytics')
            from ultralytics import YOLO

        try:
            import torch
        except ImportError:
            slicer.util.pip_install('torch')
            import torch

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
            image_array = slicer.util.arrayFromVolume(inputVolume)
            inputVolumIJKToRAS = vtk.vtkMatrix4x4()
            inputVolume.GetIJKToRASMatrix(inputVolumIJKToRAS)

            numComponents = image_array.shape[3] if len(image_array.shape) == 4 else 1

            if numComponents == 1:
                raise RuntimeError("Input image has 1 channel, expected 3-channel RGB")
            elif numComponents == 3:
                image_array = image_array.squeeze() # 
            elif numComponents == 4:
                raise RuntimeError("Input image has 4 channels (RGBA), expected 3-channel RGB")

            image_array = image_array.astype(np.uint8)

            # Run YOLO inference
            #results = model.predict(source=image_array, conf=0.25, verbose=False)

            CONF_THRES = 0.3
            IMGSZ = 640
            device = torch.device("cpu")
            results = model.predict(
                source=image_array,
                conf=CONF_THRES,
                imgsz=IMGSZ,
                device=device,
                retina_masks=True,
                augment=False,  # Disable TTA
                stream=False,   # For better memory handling
                max_det=50,      # Limit detections per image
                #verbose=False,
            )

            res = results[0]

            # Count detections
            roi_nodes = []
            parasite_count = 0
            leukocyte_count = 0
            if len(res.boxes) > 0:
                for box in res.boxes:
                    cls_id = int(box.cls[0])
                    name = None
                    if cls_id == 0:
                        parasite_count += 1
                        name = f"parasite {parasite_count}"
                    elif cls_id == 1:
                        leukocyte_count += 1
                        name = f"leukocyte {leukocyte_count}"
                    if name:
                        roi = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsROINode")
                        roi.SetName(name)
                        roi.CreateDefaultDisplayNodes()
                        roi.GetDisplayNode().SetHandlesInteractive(False)

                        # Get bounding box in RAS coordinates
                        box_xyxy = box.xyxy[0].tolist()  # [x1, y1, x2, y2]
                        corners_IJK = [
                            [box_xyxy[0], box_xyxy[1], 0, 1],  # Top-left
                            [box_xyxy[2], box_xyxy[1], 0, 1],  # Top-right
                            [box_xyxy[2], box_xyxy[3], 0, 1],  # Bottom-right
                            [box_xyxy[0], box_xyxy[3], 0, 1]   # Bottom-left
                        ]
                        boundingBox_RAS = vtk.vtkBoundingBox()
                        for corner in corners_IJK:
                            boundingBox_RAS.AddPoint(inputVolumIJKToRAS.MultiplyPoint(corner)[0:3])

                        # Compute radius with minimum size
                        boundingBoxLength_RAS = [0, 0, 0]
                        boundingBox_RAS.GetLengths(boundingBoxLength_RAS)
                        minimumRadius = 1.0
                        boundingBoxRadius_RAS = [0, 0, 0]
                        for i in range(3):
                            radius = boundingBoxLength_RAS[i] / 2
                            if radius < minimumRadius:
                                radius = minimumRadius
                            boundingBoxRadius_RAS[i] = radius

                        boundingBoxCenter_RAS = [0, 0, 0]
                        boundingBox_RAS.GetCenter(boundingBoxCenter_RAS)

                        roi.SetCenter(boundingBoxCenter_RAS)
                        roi.SetRadiusXYZ(boundingBoxRadius_RAS)

                        roi_nodes.append(roi)

            # Parasitic density
            if leukocyte_count > 0:
                density = (parasite_count / leukocyte_count) * 8000
                density_str = f"{density:.0f} parasites/µL"
            else:
                density_str = "N/A (aucun leucocyte détecté)"

            # Store results
            self.lastResults = {
                'parasite_count': parasite_count,
                'leukocyte_count': leukocyte_count,
                'parasitic_density': density_str,
                'roi_nodes': roi_nodes,
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
            #testVolume = logic.createVolumeFromArray(test_image, "TestImage")

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