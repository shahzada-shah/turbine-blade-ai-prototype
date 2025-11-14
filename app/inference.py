import uuid

from .utils.visualization import create_gradcam_heatmap

from .utils.preprocessing import load_image_from_bytes, preprocess_for_model

from .model_loader import model_loader



def decide_action(blade_type: str, severity: str, confidence: float):

    blade_type = (blade_type or "UNKNOWN").upper()



    if severity == "healthy":

        return {

            "recommended_action": "no_action",

            "needs_shutdown": False,

            "needs_investigation": False,

            "action_reason": "No visible damage detected."

        }



    if severity == "minor_damage":

        if confidence >= 0.8:

            return {

                "recommended_action": "investigate",

                "needs_shutdown": False,

                "needs_investigation": True,

                "action_reason": "Minor damage with high confidence – schedule closer inspection."

            }

        else:

            return {

                "recommended_action": "monitor",

                "needs_shutdown": False,

                "needs_investigation": False,

                "action_reason": "Possible minor damage with low confidence – monitor and recheck with clearer imagery."

            }



    if severity == "severe_damage":

        # Different thresholds for TPI vs LM if you want to be stricter

        if blade_type == "TPI" and confidence >= 0.75:

            shutdown = True

        elif blade_type == "LM" and confidence >= 0.85:

            shutdown = True

        else:

            shutdown = False



        if shutdown:

            return {

                "recommended_action": "shutdown_and_investigate",

                "needs_shutdown": True,

                "needs_investigation": True,

                "action_reason": f"High-confidence severe damage on {blade_type} blade – recommend immediate shutdown and inspection."

            }

        else:

            return {

                "recommended_action": "investigate",

                "needs_shutdown": False,

                "needs_investigation": True,

                "action_reason": f"Severe damage detected on {blade_type} blade but below shutdown confidence threshold – investigate before taking action."

            }



    # Fallback

    return {

        "recommended_action": "investigate",

        "needs_shutdown": False,

        "needs_investigation": True,

        "action_reason": "Ambiguous result – human review recommended."

    }





def analyze_blade_image(image_bytes: bytes, blade_type: str = "UNKNOWN"):

    pil_image = load_image_from_bytes(image_bytes)

    tensor = preprocess_for_model(pil_image)



    # REAL prediction now

    severity, confidence = model_loader.predict(tensor)

    damage_detected = severity != "healthy"

    confidence = round(float(confidence), 2)

    # Get class index for Grad-CAM
    target_class_idx = None
    if model_loader.model is not None and model_loader.class_names:
        try:
            target_class_idx = model_loader.class_names.index(severity)
        except ValueError:
            target_class_idx = None

    heatmap_filename = f"results/heatmap_{uuid.uuid4().hex}.png"

    # Create real Grad-CAM heatmap (falls back to dummy if model not available)
    create_gradcam_heatmap(
        model_loader.model,
        tensor,
        pil_image,
        heatmap_filename,
        target_class_idx=target_class_idx
    )



    action_info = decide_action(blade_type, severity, confidence)



    return {

        "blade_type": blade_type,

        "damage_detected": damage_detected,

        "severity": severity,

        "confidence": confidence,

        "heatmap_path": heatmap_filename,

        **action_info,

    }
