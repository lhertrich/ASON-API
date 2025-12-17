import qupath.lib.io.GsonTools
import qupath.lib.regions.RegionRequest
import qupath.imagej.tools.IJTools
import qupath.fx.dialogs.Dialogs
import java.io.ByteArrayOutputStream
import javax.imageio.ImageIO

// --- CONFIGURATION ---
def serverUrl = "http://127.0.0.1:8000"

// --- GUI: SELECT TASK ---
def tasks = ["Segment Nuclei", "Segment Tissue", "Nuclei Graph", "Layers", "Layers with Orientations", "Alignment Overlay", "Unorganized Regions"]
def task = Dialogs.showChoiceDialog("Select Task", "Choose a pipeline step", tasks, "Segment Tissue")
if (task == null) return

def probTh = 0.5
def nmsTh = 0.3

if (task == "Segment Nuclei") {
    def pInput = Dialogs.showInputDialog("Probability Threshold", "Enter threshold (0-1)", 0.5)
    if (pInput == null) return
    probTh = pInput.doubleValue()
    
    def nInput = Dialogs.showInputDialog("NMS Threshold", "Enter NMS threshold (0-1)", 0.3)
    if (nInput == null) return
    nmsTh = nInput.doubleValue()
}

// --- PREPARE DATA ---
def imageData = getQuPath().getImageData()
def server = imageData.getServer()

// FIX: Correct way to get the region in 0.6.0
def selectedPathObject = getSelectedObject()
def region

if (selectedPathObject != null && selectedPathObject.hasROI()) {
    // Use the selected annotation as the region
    region = RegionRequest.create(server.getPath(), 1.0, selectedPathObject.getROI())
    println "Processing selected ROI..."
} else {
    // Use the full image at full resolution (downsample 1.0)
    region = RegionRequest.create(server, 1.0)
    println "No selection found. Processing full image..."
}

// Export region as PNG bytes
def img = IJTools.convertToImagePlus(imageData, region).getBufferedImage()
def baos = new ByteArrayOutputStream()
ImageIO.write(img, "png", baos)
def bytes = baos.toByteArray()

// --- API CALL ---
def taskParam = task.toLowerCase().replace(' ', '_')
def urlString = "${serverUrl}/segment?task=${taskParam}&prob_th=${probTh}&nms_th=${nmsTh}"

try {
    def url = new URL(urlString)
    def connection = url.openConnection() as HttpURLConnection
    connection.setRequestMethod("POST")
    connection.setRequestProperty("Content-Type", "image/png")
    connection.setDoOutput(true)
    connection.getOutputStream().write(bytes)

    if (connection.getResponseCode() == 200) {
        def jsonResponse = connection.getInputStream().getText()
        
        // Use GsonTools to parse the GeoJSON string
        def objects = GsonTools.readObjects(jsonResponse)
        
        if (objects.isEmpty()) {
            println "No objects detected."
        } else {
            // Apply the coordinate transformation
            def transform = region.getGeometryTransform()
            objects.each { it.getROI().transform(transform) }
            
            addObjects(objects)
            fireHierarchyUpdate()
            println "Successfully imported ${objects.size()} objects."
        }
    } else {
        println "Server Error: " + connection.getResponseCode()
    }
} catch (Exception e) {
    println "Connection failed: " + e.getMessage()
}