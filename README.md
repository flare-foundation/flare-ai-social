# Flare AI Social

Flare AI Kit template for Social AI Agents.

## 🏗️ Build & Run Instructions

### Fine tuning a model over a dataset

1. **Prepare the Environment File:**  
   Rename `.env.example` to `.env` and update the variables accordingly.
   Some parameters are specific to model fine-tuning:

   | Parameter             | Description                                                                | Default                              |
   | --------------------- | -------------------------------------------------------------------------- | ------------------------------------ |
   | `tuned_model_name`    | Name of the newly tuned model.                                             | `pugo-hilion`                        |
   | `tuning_source_model` | Name of the foundational model to tune on.                                 | `models/gemini-1.5-flash-001-tuning` |
   | `epoch_count`         | Number of tuning epochs to run. An epoch is a pass over the whole dataset. | `100`                                |
   | `batch_size`          | Number of examples to use in each training batch.                          | `4`                                  |
   | `learning_rate`       | Step size multiplier for the gradient updates.                             | `0.001`                              |

2. **Prepare a dataset:**
   An example dataset is provided in `src/data/training_data.json`, which consists of tweet from
   [Hugo Philion's X](https://x.com/HugoPhilion) account. You can use any publicly available dataset
   for model fine-tuning.

3. **Tune a new model**
   Set the name of the new tuned model in `src/flare_ai_social/tune_model.py`, then:

   ```bash
   uv run start-tuning
   ```

4. **Observe loss parameters:**
   After tuning in complete, a training loss PNG will be saved in the root folder corresponding to the new model.
   Ideally the loss should minimize to near 0 after several training epochs.

5. **Test the new model**
   Select the new tuned model and test it against a set of prompts:

   ```bash
   uv run start-social
   ```
