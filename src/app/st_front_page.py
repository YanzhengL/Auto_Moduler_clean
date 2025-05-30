import os
import streamlit as st
import time
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


def real_work(user_request):
    """Simulate a time-consuming backend process synchronously."""
    with st.status("Processing your request...", expanded=True) as status:
        status.update(label="Real work running...", state="running")

        time.sleep(3)  # Simulate work (e.g., generating a PLECS model)
        BASE_DIR = os.path.dirname(os.path.dirname(__file__))  # one level up from /app
        image_path = os.path.join(BASE_DIR, "resources", "boost.png")
        #image_path = os.path.join("src", "resources", "boost.png")
        print("Loading image from:", image_path)
        st.image(image_path, caption="Boost Converter Design", use_container_width=True)

        status.update(label=f"P✅ Model successfully generated!", state="complete")
        time.sleep(1)
        status.update(label=f"Simulation running", state="running")
        time.sleep(3)
        image_path = os.path.join(BASE_DIR, "resources", "plot.png")
        st.image(image_path, caption="Simulation results", use_container_width=True)
        status.update(label=f"P✅ Model simulation process done!", state="complete")
        # Store result in session state
        st.session_state.processing_result = f"Processing completed for: {user_request}"
        st.session_state.processing_complete = True  # Mark as done


def get_backend_response(user_request):
    """Check user request and process it synchronously."""
    keywords = ["bulk converter", "boost converter", "converter"]

    if any(keyword in user_request.lower() for keyword in keywords):
        st.session_state.processing_complete = False  # Reset flag
        real_work(user_request)  # Run synchronously
        return {"status": "success", "message": "Processing started..."}
    else:
        return {"status": "error",
                "message": "I'm not sure how to respond to that. I can only generate bulk converters and boost converters."}


# Initialize session state variables
if "processing_result" not in st.session_state:
    st.session_state.processing_result = ""
if "processing_complete" not in st.session_state:
    st.session_state.processing_complete = True  # Default to True to avoid showing "working" on first load

# Streamlit UI
st.title("🚀 AutoModular - AI+Engineering")
st.info("Circuit Model Generator developed by the Aviation Electrification Research Team at the University of Nottingham.")

# Collect user request input
st.markdown("---")  # Separator for clarity
user_request = st.text_area("Enter your model description or request:", key="user_request")

if st.button("Analyze and Generate Your MATLAB Model"):
    if user_request:
        response = get_backend_response(user_request)
        if response["status"] == "error":
            st.session_state.processing_result = response["message"]
            st.session_state.processing_complete = True
            st.rerun()
    else:
        st.warning("Please enter a request to proceed.")

# Display final result
if st.session_state.processing_complete and st.session_state.processing_result:
    st.success(st.session_state.processing_result)