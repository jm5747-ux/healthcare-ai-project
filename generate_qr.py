import qrcode
from PIL import Image

# Your Streamlit app URL
streamlit_url = "https://healthcare-ai-project-jm5747-ux.streamlit.app"

# Create QR code
qr = qrcode.QRCode(
    version=1,
    error_correction=qrcode.constants.ERROR_CORRECT_L,
    box_size=10,
    border=4,
)

# Add data to QR code
qr.add_data(streamlit_url)
qr.make(fit=True)

# Create image
img = qr.make_image(fill_color="black", back_color="white")

# Save the QR code
img.save("streamlit_app_qr_code.png")

print("✅ QR code generated successfully!")
print(f"📱 URL: {streamlit_url}")
print("🖼️  Saved as: streamlit_app_qr_code.png")
print("\nYou can now:")
print("1. Share the QR code image")
print("2. Print it for easy access")
print("3. Use it in presentations or documents")
