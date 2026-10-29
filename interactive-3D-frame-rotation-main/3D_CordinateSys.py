"""
Rigid Body Rotation Visualizer (3D)
===================================

Interactive tool that shows:
  * a FIXED reference frame (solid arrows: X=red, Y=green, Z=blue)
  * a ROTATABLE body frame (dashed arrows), driven in real time by sliders
  * the live 3x3 rotation matrix mapping body-frame coordinates
    into the reference frame

Run:  python rigid_body_rotation_3d.py
Requires: numpy, matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, RadioButtons, Button

# ----------------------------------------------------------------------
# Rotation math
# ----------------------------------------------------------------------
def calculateRotationMatrixX(angleInRadians):
    cosineOfAngle = np.cos(angleInRadians)
    sineOfAngle = np.sin(angleInRadians)
    return np.array([[1, 0, 0],
                     [0, cosineOfAngle, -sineOfAngle],
                     [0, sineOfAngle, cosineOfAngle]])

def calculateRotationMatrixY(angleInRadians):
    cosineOfAngle = np.cos(angleInRadians)
    sineOfAngle = np.sin(angleInRadians)
    return np.array([[cosineOfAngle, 0, sineOfAngle],
                     [0, 1, 0],
                     [-sineOfAngle, 0, cosineOfAngle]])

def calculateRotationMatrixZ(angleInRadians):
    cosineOfAngle = np.cos(angleInRadians)
    sineOfAngle = np.sin(angleInRadians)
    return np.array([[cosineOfAngle, -sineOfAngle, 0],
                     [sineOfAngle, cosineOfAngle, 0],
                     [0, 0, 1]])

# ----------------------------------------------------------------------
# Figure layout
# ----------------------------------------------------------------------
mainFigure = plt.figure(figsize=(12, 8))
mainFigure.canvas.manager.set_window_title("Rigid Body Rotation Visualizer")
threeDimensionalAxes = mainFigure.add_axes([0.02, 0.25, 0.68, 0.72], projection="3d")

axisColors = {"X": "#d62728", "Y": "#2ca02c", "Z": "#1f77b4"}
noRotationMatrix = np.eye(3)

def drawCoordinateFrames(plotAxes, currentRotationMatrix):
    """Draw the fixed frame (solid) and the rotated body frame (dashed)."""
    plotAxes.cla()
    plotAxes.set_xlim(-1.4, 1.4)
    plotAxes.set_ylim(-1.4, 1.4)
    plotAxes.set_zlim(-1.4, 1.4)
    plotAxes.set_box_aspect((1, 1, 1))
    plotAxes.set_xlabel("X")
    plotAxes.set_ylabel("Y")
    plotAxes.set_zlabel("Z")
    plotAxes.set_title("Fixed frame (solid)  vs.  Body frame (dashed)")

    for axisIndex, (axisName, axisColor) in enumerate(axisColors.items()):
        globalBasisVector = noRotationMatrix[:, axisIndex]
        localBasisVector = currentRotationMatrix @ globalBasisVector

        # Draw fixed reference frame
        plotAxes.quiver(0, 0, 0, globalBasisVector[0], globalBasisVector[1], globalBasisVector[2],
                  color=axisColor, linewidth=2.5, arrow_length_ratio=0.12)
        plotAxes.text(*(globalBasisVector * 1.15), axisName, color=axisColor, fontsize=12, weight="bold")

        # Draw rotatable body frame
        plotAxes.quiver(0, 0, 0, localBasisVector[0], localBasisVector[1], localBasisVector[2],
                  color=axisColor, linewidth=2.5, linestyle="--",
                  arrow_length_ratio=0.12, alpha=0.9)
        plotAxes.text(*(localBasisVector * 1.28), axisName + "'", color=axisColor, fontsize=12,
                style="italic", alpha=0.9)

    # Faint unit sphere wireframe for spatial context
    azimuthAngle, elevationAngle = np.mgrid[0:2 * np.pi:24j, 0:np.pi:16j]
    plotAxes.plot_wireframe(np.cos(azimuthAngle) * np.sin(elevationAngle), np.sin(azimuthAngle) * np.sin(elevationAngle), np.cos(elevationAngle),
                      color="gray", alpha=0.12, linewidth=0.5)

# ----------------------------------------------------------------------
# Widgets
# ----------------------------------------------------------------------
sliderPlotAreas = {
    "X": mainFigure.add_axes([0.76, 0.72, 0.20, 0.03]),
    "Y": mainFigure.add_axes([0.76, 0.62, 0.20, 0.03]),
    "Z": mainFigure.add_axes([0.76, 0.52, 0.20, 0.03]),
}
angleSliders = {
    axisName: Slider(sliderPlotAreas[axisName], f"Input {axisName} (deg)",
                 -180.0, 180.0, valinit=0.0, valstep=1.0,
                 color=axisColors[axisName])
    for axisName in ("X", "Y", "Z")
}

frameSelectionArea = mainFigure.add_axes([0.76, 0.34, 0.20, 0.13])
frameSelectionArea.set_title("Rotation axes", fontsize=10)
frameSelectionRadioButtons = RadioButtons(frameSelectionArea,
                           ("Global (fixed axes)", "Local (body axes)"),
                           active=0)

stepSelectionArea = mainFigure.add_axes([0.76, 0.17, 0.20, 0.14])
stepSelectionArea.set_title("Angle step", fontsize=10)
stepSelectionRadioButtons = RadioButtons(stepSelectionArea, ("1 deg", "10 deg", "90 deg"), active=0)

resetButtonArea = mainFigure.add_axes([0.76, 0.09, 0.20, 0.05])
resetButton = Button(resetButtonArea, "Reset")

rotationMatrixDisplayText = mainFigure.text(0.74, 0.98, "", family="monospace", fontsize=10,
                       verticalalignment="top")

# ----------------------------------------------------------------------
# Stateful Update logic
# ----------------------------------------------------------------------
currentCumulativeRotationMatrix = np.eye(3)
previousSliderAngles = {"X": 0.0, "Y": 0.0, "Z": 0.0}

def getCurrentSliderAngles():
    return (angleSliders["X"].val, angleSliders["Y"].val, angleSliders["Z"].val)

def updateVisualization(eventTrigger=None):
    global currentCumulativeRotationMatrix
    xRotationAngle, yRotationAngle, zRotationAngle = getCurrentSliderAngles()
    selectedRotationMode = frameSelectionRadioButtons.value_selected

    # Calculate exactly how much the slider changed since last update
    changeInXAngle = xRotationAngle - previousSliderAngles["X"]
    changeInYAngle = yRotationAngle - previousSliderAngles["Y"]
    changeInZAngle = zRotationAngle - previousSliderAngles["Z"]

    # Update state trackers
    previousSliderAngles["X"] = xRotationAngle
    previousSliderAngles["Y"] = yRotationAngle
    previousSliderAngles["Z"] = zRotationAngle

    # Only process matrix math if a drag actually occurred
    if changeInXAngle != 0 or changeInYAngle != 0 or changeInZAngle != 0:
        incrementalRotationX = calculateRotationMatrixX(np.radians(changeInXAngle))
        incrementalRotationY = calculateRotationMatrixY(np.radians(changeInYAngle))
        incrementalRotationZ = calculateRotationMatrixZ(np.radians(changeInZAngle))

        if selectedRotationMode.startswith("Local"):
            # Post-multiply to rotate around the CURRENT local axes
            currentCumulativeRotationMatrix = currentCumulativeRotationMatrix @ incrementalRotationX @ incrementalRotationY @ incrementalRotationZ
        else:
            # Pre-multiply to rotate around the FIXED global axes
            currentCumulativeRotationMatrix = incrementalRotationZ @ incrementalRotationY @ incrementalRotationX @ currentCumulativeRotationMatrix

    drawCoordinateFrames(threeDimensionalAxes, currentCumulativeRotationMatrix)

    calculationFormula = "Post-multiply (Incremental)" if selectedRotationMode.startswith("Local") else "Pre-multiply (Incremental)"

    matrixString = (
        f"Rotation matrix  {calculationFormula}\n"
        f"mode: {selectedRotationMode}\n"
        f"Cumul. Input: X={xRotationAngle:5.1f} Y={yRotationAngle:5.1f} Z={zRotationAngle:5.1f}\n\n"
        f"[{currentCumulativeRotationMatrix[0, 0]: 7.3f} {currentCumulativeRotationMatrix[0, 1]: 7.3f} {currentCumulativeRotationMatrix[0, 2]: 7.3f}]\n"
        f"[{currentCumulativeRotationMatrix[1, 0]: 7.3f} {currentCumulativeRotationMatrix[1, 1]: 7.3f} {currentCumulativeRotationMatrix[1, 2]: 7.3f}]\n"
        f"[{currentCumulativeRotationMatrix[2, 0]: 7.3f} {currentCumulativeRotationMatrix[2, 1]: 7.3f} {currentCumulativeRotationMatrix[2, 2]: 7.3f}]"
    )
    rotationMatrixDisplayText.set_text(matrixString)
    mainFigure.canvas.draw_idle()


def handleStepChange(selectedLabel):
    stepValue = float(selectedLabel.split()[0])
    for singleSlider in angleSliders.values():
        singleSlider.valstep = stepValue
        singleSlider.set_val(round(singleSlider.val / stepValue) * stepValue)

def resetAllValues(eventTrigger):
    global currentCumulativeRotationMatrix
    currentCumulativeRotationMatrix = np.eye(3)

    # Safely detach events, reset sliders, and reattach to avoid double-firing deltas
    for axisName, singleSlider in angleSliders.items():
        singleSlider.eventson = False
        singleSlider.reset()
        previousSliderAngles[axisName] = 0.0
        singleSlider.eventson = True

    updateVisualization()


for singleSlider in angleSliders.values():
    singleSlider.on_changed(updateVisualization)

stepSelectionRadioButtons.on_clicked(handleStepChange)
frameSelectionRadioButtons.on_clicked(updateVisualization)
resetButton.on_clicked(resetAllValues)

updateVisualization()
plt.show()
