import json
import os
from datetime import datetime
from typing import Optional


def log_simple_feedback(
    question: str, answer: str, feedback_type: str, comment: Optional[str] = None, user_id: str = "user_1"
):
    """Super simple feedback logging that definitely works"""
    log_dir = "./logs/feedback"
    os.makedirs(log_dir, exist_ok=True)

    log_file = os.path.join(log_dir, f"feedback_log_{datetime.now().strftime('%Y%m%d')}.jsonl")

    entry = {
        "timestamp": datetime.now().isoformat(),
        "user_id": user_id,
        "question": question,
        "answer_snippet": answer[:100] + "..." if len(answer) > 100 else answer,
        "feedback_type": feedback_type,
        "comment": comment,
    }

    try:
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            f.flush()
        print(f"✅ SUCCESS: Logged {feedback_type} feedback to {log_file}")
        return True
    except Exception as e:
        print(f"❌ ERROR: Failed to log feedback: {e}")
        return False


def render_simple_feedback_buttons(st, question: str, answer: str, user_id: str = "user_1"):
    """Render the simplest possible feedback buttons"""

    # Create a simple unique key based on question
    feedback_key = f"feedback_{hash(question)}"

    st.markdown("---")
    st.markdown("### 📝 Rate this response")

    # Check if already given feedback
    if st.session_state.get(f"{feedback_key}_done", False):
        st.success("✅ Thank you! Feedback recorded.")
        return

    col1, col2 = st.columns(2)

    with col1:
        if st.button("👍 GOOD", key=f"{feedback_key}_pos", use_container_width=True):
            print("🔥 POSITIVE BUTTON CLICKED!")
            success = log_simple_feedback(question, answer, "positive", None, user_id)
            if success:
                st.session_state[f"{feedback_key}_done"] = True
                st.balloons()
                st.success("✅ Positive feedback saved!")
                st.rerun()
            else:
                st.error("❌ Failed to save feedback")

    with col2:
        if st.button("👎 BAD", key=f"{feedback_key}_neg", use_container_width=True):
            print("🔥 NEGATIVE BUTTON CLICKED!")
            success = log_simple_feedback(question, answer, "negative", None, user_id)
            if success:
                st.session_state[f"{feedback_key}_done"] = True
                st.success("✅ Negative feedback saved!")
                st.rerun()
            else:
                st.error("❌ Failed to save feedback")
