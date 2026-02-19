import streamlit as st

st.set_page_config(page_title="Secret Test", page_icon="🔑")
st.title("🔑 Streamlit Secrets Diagnostic")

st.write("---")

# Test 1: Does st.secrets exist?
st.write("### Test 1: st.secrets exists?")
try:
    secrets_exist = hasattr(st, 'secrets')
    st.write(f"Result: {secrets_exist}")
except Exception as e:
    st.error(f"Error: {e}")

# Test 2: What keys are available?
st.write("### Test 2: Available keys")
try:
    keys = list(st.secrets.keys())
    st.write(f"Keys found: {keys}")
except Exception as e:
    st.error(f"Error: {e}")

# Test 3: Try each key
st.write("### Test 3: Access each secret")

test_keys = [
    "HUGGINGFACE_API_KEY",
    "OPENAI_API_KEY",
    "XAI_API_KEY", 
    "MODEL_PASSWORD",
    "ADMIN_PASSWORD"
]

for key in test_keys:
    try:
        val = st.secrets[key]
        preview = str(val)[:8] + "..." if len(str(val)) > 8 else str(val)
        st.success(f"✅ {key} = `{preview}` (length: {len(str(val))})")
    except KeyError:
        st.error(f"❌ {key} = NOT FOUND (KeyError)")
    except Exception as e:
        st.error(f"❌ {key} = Error: {e}")

# Test 4: Show raw structure
st.write("### Test 4: Raw structure")
try:
    for key in st.secrets:
        val = st.secrets[key]
        val_type = type(val).__name__
        st.write(f"- `{key}`: type={val_type}")
except Exception as e:
    st.error(f"Error: {e}")

st.write("---")
st.info("Copy the output above and share it!")
