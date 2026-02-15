# Model Performance Comparison

| Model                                  |   Accuracy (%) |   Hallucination Rate (%) |   Confident Hallucination (%) |   False Negative Rate (%) |   Total Hallucinations |   Total False Negatives |
|:---------------------------------------|---------------:|-------------------------:|------------------------------:|--------------------------:|-----------------------:|------------------------:|
| Llama-4-Maverick-17B-128E-Instruct-FP8 |          100   |                        0 |                             0 |                         0 |                      0 |                       0 |
| gpt-oss-120b                           |          100   |                        0 |                             0 |                         0 |                      0 |                       0 |
| Qwen3-Next-80B-A3B-Instruct            |          100   |                        0 |                             0 |                         0 |                      0 |                       0 |
| Kimi-K2.5                              |          100   |                        0 |                             0 |                         0 |                      0 |                       0 |
| GLM-5                                  |          100   |                        0 |                             0 |                         0 |                      0 |                       0 |
| Mixtral-8x7B-Instruct-v0.1             |           97.5 |                        0 |                             0 |                         5 |                      0 |                       1 |
| Mistral-Small-24B-Instruct-2501        |           95   |                        0 |                             0 |                        10 |                      0 |                       2 |

## Key Metrics Explanation

- **Accuracy**: Percentage of correctly classified statements
- **Hallucination Rate**: Percentage of fake facts accepted as true
- **Confident Hallucination**: Hallucinations with high confidence
- **False Negative Rate**: Percentage of real facts rejected
