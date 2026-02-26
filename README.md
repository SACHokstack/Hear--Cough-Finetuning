# HeAR-TB: TB Cough Screening with Google HeAR

**Domain-Aware Dual Heads Model**  
(Entry for MedGemma Impact Challenge)

This project uses Google's **HeAR** model (frozen) to detect tuberculosis from cough sounds.

### What we did (the novel part)

Instead of training one model on all coughs, we made two separate "experts":

- One expert only learns from **natural coughs** (passive, everyday coughing)
- One expert only learns from **forced coughs** (when people are asked to cough hard)

Then we average their opinions for each patient.

This makes the model understand that natural and forced coughs sound different — and still gives one final TB risk score per person.

### Results (patient-level, repeated 5-fold CV)

| Metric            | Value              |
|-------------------|--------------------|
| Patient AUC       | **0.7476 ± 0.0932** |
| Sensitivity       | ~0.78–0.83         |
| Accuracy          | 0.728 ± 0.060      |

### Model
The final trained model can be found here https://huggingface.co/sach3v/Domain_aware_dual_head_HEar
### Dataset 
The model was trained on a wide variety of data with a 33k sample of coughs divided by forced and passive , the dataset can be found here https://zenodo.org/records/10431329

### Quick start (how to use it)

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/hear-tb-domain-aware.git
cd hear-tb-domain-aware

# 2. Install requirements
pip install -r requirements.txt

# 3. Run the demo (needs HeAR model loaded)
python demo.py


