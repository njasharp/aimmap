import os
import math
import streamlit as st
from typing import Dict, Optional
from groq import Groq
import cairosvg
import re

# --------------------------------------------------------------------
# Streamlit page configuration
# --------------------------------------------------------------------
st.set_page_config(
    layout="wide",
    page_title="AI Mind Map Generator",
    initial_sidebar_state="expanded"
)

# --------------------------------------------------------------------
# Helper Functions for Mermaid Parsing and SVG Generation
# --------------------------------------------------------------------
def parse_mermaid_to_svg(mermaid_code, layout="flowchart"):
    """
    Parses Mermaid code to extract nodes and edges and generates SVG elements based on the chosen layout.

    :param mermaid_code: Mermaid graph syntax as a string.
    :param layout: The desired layout type (e.g., "wireframe" or "flowchart").
    :return: SVG content as a string.
    """
    nodes = {}
    edges = []

    # Extract nodes
    node_pattern = re.compile(r'(\w+)\[(.*?)\]')
    for match in node_pattern.finditer(mermaid_code):
        node_id, label = match.groups()
        nodes[node_id] = label

    # Extract edges
    edge_pattern = re.compile(r'(\w+)\s*-->\s*(\w+)')
    for match in edge_pattern.finditer(mermaid_code):
        source, target = match.groups()
        edges.append((source, target))

    # Initialize SVG content
    svg_content = '''
    <svg viewBox="0 0 1200 800" xmlns="http://www.w3.org/2000/svg">
        <!-- Background -->
        <rect width="1200" height="800" fill="#ffffff"/>
    '''

    # Layout logic
    if layout == "flowchart":
        columns = 4  # Default columns for flowchart
        spacing_x = 250
        spacing_y = 150
        start_x = 100
        start_y = 100

        node_positions = {}

        for i, (node_id, label) in enumerate(nodes.items()):
            x = start_x + (i % columns) * spacing_x
            y = start_y + (i // columns) * spacing_y
            node_positions[node_id] = (x, y)

            svg_content += f'''
            <g>
                <rect x="{x - 75}" y="{y - 25}" width="150" height="50" fill="#4CAF50" rx="10" ry="10"/>
                <text x="{x}" y="{y + 5}" text-anchor="middle" fill="white" font-family="Arial" font-size="14">{label}</text>
            </g>
            '''

        # Draw edges
        for source, target in edges:
            if source in node_positions and target in node_positions:
                x1, y1 = node_positions[source]
                x2, y2 = node_positions[target]
                svg_content += f'''
                <line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="#000000" stroke-width="2" marker-end="url(#arrowhead)"/>
                '''
    elif layout == "wireframe":
        center_x, center_y = 600, 400
        radius = max(150, min(300, 250 + 10 * len(nodes)))
        angle_step = 2 * math.pi / max(1, len(nodes))  # Avoid division by zero

        node_positions = {}
        for i, (node_id, label) in enumerate(nodes.items()):
            angle = i * angle_step
            x = center_x + radius * math.cos(angle)
            y = center_y + radius * math.sin(angle)
            node_positions[node_id] = (x, y)

            svg_content += f'''
            <g>
                <rect x="{x - 75}" y="{y - 25}" width="150" height="50" fill="#f39c12" rx="10" ry="10"/>
                <text x="{x}" y="{y + 5}" text-anchor="middle" fill="black" font-family="Arial" font-size="14">{label}</text>
            </g>
            '''

        # Draw edges
        for source, target in edges:
            if source in node_positions and target in node_positions:
                x1, y1 = node_positions[source]
                x2, y2 = node_positions[target]
                svg_content += f'''
                <line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="#d35400" stroke-width="2" marker-end="url(#arrowhead)"/>
                '''

    # Add arrowhead for edges
    svg_content += '''
        <defs>
            <marker id="arrowhead" markerWidth="10" markerHeight="7" refX="10" refY="3.5" orient="auto">
                <polygon points="0 0, 10 3.5, 0 7" fill="black"/>
            </marker>
        </defs>
    '''

    svg_content += '</svg>'
    return svg_content

def generate_and_save_final_image(mermaid_code, layout="flowchart"):
    """
    Generates the final image based on Mermaid code and saves it as a PNG file.

    :param mermaid_code: Mermaid graph syntax as a string.
    :param layout: The desired layout type (e.g., "wireframe" or "flowchart").
    :return: Tuple (success, output_file or error message).
    """
    try:
        # Parse Mermaid to SVG
        svg_content = parse_mermaid_to_svg(mermaid_code, layout)
        
        # Ensure the output directory exists
        output_directory = "images"
        os.makedirs(output_directory, exist_ok=True)
        
        # Define output file path
        output_file = os.path.join(output_directory, f"final_image_{layout}.png")
        
        # Convert SVG to PNG and save
        cairosvg.svg2png(bytestring=svg_content.encode("utf-8"), write_to=output_file)
        return True, output_file
    except Exception as e:
        return False, str(e)

# --------------------------------------------------------------------
# Supported Models
# --------------------------------------------------------------------
SUPPORTED_MODELS: Dict[str, str] = {
    "Llama 3 8B": "llama3-8b-8192",
    "Llama 3.2 1B (Preview)": "llama-3.2-1b-preview",
    "Llama 3 70B": "llama3-70b-8192",
    "Mixtral 8x7B": "mixtral-8x7b-32768",
    "Gemma 2 9B": "gemma2-9b-it",
    "Llama 3.2 11B Vision (Preview)": "llama-3.2-11b-vision-preview",
    "Llama 3.2 11B Text (Preview)": "llama-3.2-11b-text-preview",
    "Llama 3.1 8B Instant (Text-Only Workloads)": "llama-3.1-8b-instant",
    "Llama 3.2 90B Vision (Preview)": "llama-3.2-90b-vision-preview",
    "Llama 3.1 70B Versatile": "llama-3.1-70b-versatile",
    "Llama 3.1 8B Instant": "llama-3.1-8b-instant",
    "Llama 3.2 11B Vision (Preview)": "llama-3.2-11b-vision-preview",
    "Llama 3.2 1B (Preview)": "llama-3.2-1b-preview",
    "Llama 3.2 3B (Preview)": "llama-3.2-3b-preview",
    "Llama 3.2 90B Vision (Preview)": "llama-3.2-90b-vision-preview",
    "Llama 3.3 70B SpecDec": "llama-3.3-70b-specdec",
    "Llama 3.3 70B Versatile": "llama-3.3-70b-versatile",
}

MAX_TOKENS: int = 1500

# --------------------------------------------------------------------
# Initialize Groq client with API key
# --------------------------------------------------------------------
@st.cache_resource
def get_groq_client() -> Optional[Groq]:
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        st.error("GROQ_API_KEY not found in environment variables. Please set it and restart the app.")
        return None
    return Groq(api_key=groq_api_key)

client = get_groq_client()

# --------------------------------------------------------------------
# SIDEBAR
# --------------------------------------------------------------------
st.sidebar.image("icon.png", width=300)
st.sidebar.title("Model Configuration")

selected_model = st.sidebar.selectbox("Choose an AI Model", list(SUPPORTED_MODELS.keys()))

st.sidebar.subheader("Temperature")
temperature = st.sidebar.slider(
    "Set temperature for generation variability:", 
    min_value=0.0, 
    max_value=1.0, 
    value=0.7
)

# Add layout selection
st.sidebar.subheader("Layout Configuration")
layout = st.sidebar.radio(
    "Select the layout for the mind map:",
    options=["flowchart", "wireframe"]
)

# --------------------------------------------------------------------
# MAIN CONTENT
# --------------------------------------------------------------------
st.title("AI Mind Map Generator")
st.markdown(
    """
    Enter your concepts or a short description below, then click **Generate Mind Map**. 
    The Groq LLM will produce Mermaid diagram code, which we'll display below.
    """
)

# Text area for user input
mind_map_prompt = st.text_area(
    "Describe your mind map focus:",
    placeholder="e.g. 'Attention and Intention in personal development'"
)

if st.button("Generate Mind Map"):
    if not mind_map_prompt.strip():
        st.warning("Please provide a description or concept for the mind map.")
    elif client:
        with st.spinner("Generating your mind map..."):
            prompt = f"""
            You are an AI that generates a Mind Map in Mermaid format. 
            The user wants a mind map about: {mind_map_prompt}.
            Please output ONLY the Mermaid diagram, nothing else.
            """

            try:
                response = client.chat.completions.create(
                    model=SUPPORTED_MODELS[selected_model],
                    messages=[
                        {"role": "system", "content": "You are an AI that generates mind maps in Mermaid code."},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=temperature,
                    max_tokens=MAX_TOKENS,
                )

                mermaid_code = response.choices[0].message.content.strip()

                st.subheader("Generated Mind Map")
                st.markdown(
                    f"""
                    ```mermaid
                    {mermaid_code}
                    ```
                    """,
                    unsafe_allow_html=True
                )

                st.download_button(
                    label="Download Mermaid Code",
                    data=mermaid_code,
                    file_name="mind_map_mermaid.txt",
                    mime="text/plain"
                )

                # Generate and display the final image based on layout
                success, result = generate_and_save_final_image(mermaid_code, layout)
                if success:
                    st.image(result, caption=f"Generated Mind Map ({layout.capitalize()} Layout)", use_column_width=True)
                else:
                    st.error(f"Failed to generate image: {result}")

            except Exception as e:
                st.error(f"Error generating mind map: {e}")
    else:
        st.error("Groq client not initialized. Make sure you have set your GROQ_API_KEY environment variable.")

st.info("Built by dw — This app uses Groq LLM to generate Mermaid-based mind maps.")

