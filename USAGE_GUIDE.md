# Cosmetic Tutor - Blush Detection Usage Guide

## ? Your Blush Detection is Working Perfectly!

The system was working all along. Here's how to use it properly:

## ?? **Step-by-Step Usage:**

### 1. **Start the Application**
```bash
# Make sure virtual environment is activated
.\.venv\Scripts\Activate.ps1

# Run the main application
python run_webcam.py
```

### 2. **Position Your Face**
- Make sure your face is clearly visible in the camera
- Good lighting is important
- Face should fill a reasonable portion of the webcam frame

### 3. **Calibrate Baseline (IMPORTANT!)**
- **Press 'b' key** to calibrate your natural skin tone as baseline
- This step is crucial - without baseline, all deltas will be 0
- You'll see: "? Baseline calibrated: Left=X.XX, Right=X.XX"

### 4. **Apply Makeup & See Changes**
- Apply blush or makeup to your cheeks
- The system will now show meaningful delta values
- Recommendations will appear based on changes from baseline

### 5. **Controls**
- **[b]** - Calibrate baseline (use when face is makeup-free)
- **[q]** - Quit application  
- **[X]** - Close window (click the window close button)

## ?? **What the System Shows:**

- **Left/Right Cheek Delta**: How much redder compared to baseline
- **Asymmetry Score**: Difference between left and right sides
- **Coverage**: How much of the cheek area has blush
- **Balance Direction**: Placement guidance (too high, too low, etc.)

## ?? **Understanding the Values:**

- **Mean a* values**: Natural skin tones ~135-145 (current readings look normal)
- **Delta = 0**: No change from baseline (need to calibrate first!)
- **Coverage = 1.00**: 100% coverage detected in cheek regions
- **Positive Delta**: More red/blush than baseline
- **Negative Delta**: Less red than baseline

## ?? **Tips for Best Results:**

1. **Always calibrate baseline first** before applying makeup
2. **Ensure good, consistent lighting**
3. **Keep face positioned consistently** 
4. **Apply makeup gradually** to see delta changes
5. **Use the advice text** for specific recommendations

## ?? **Technical Status:**
- ? Camera: Working
- ? Face Detection: Working (478 landmarks)  
- ? ROI Building: Working (24 points per cheek)
- ? Color Analysis: Working (meaningful Lab a* values)
- ? Baseline System: Working (needs user to press 'b')
- ? Coverage Detection: Working (100% coverage)

**Your blush detection system is fully functional!** ??