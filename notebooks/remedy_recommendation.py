from remedies import REMEDIES

# Example disease (replace with model prediction later)
predicted_disease = "Pepper__bell___healthy"

if predicted_disease in REMEDIES:
    print("Disease:", predicted_disease)

    print("\nOrganic Remedies:")
    for r in REMEDIES[predicted_disease]["organic"]:
        print("-", r)

    print("\nChemical Remedies:")
    for r in REMEDIES[predicted_disease]["chemical"]:
        print("-", r)
else:
    print("No remedy information available for this disease.")
