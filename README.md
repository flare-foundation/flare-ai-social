# 🤖 Flare AI Social

Flare AI Kit template for building AI-powered social media agents that interact with the Flare Network community through secure model fine-tuning and deployment.

## 📋 Overview

TODO: Add detailed project overview.

## 🏗️ Project Structure

The Flare AI Social template provides a modular architecture for building AI-powered social media agents. Here's a breakdown of the core files and their purposes:

```
src/flare_ai_social/
├── ai/                     # AI Provider implementations
│   ├── __init__.py        # Package exports
│   ├── base.py            # Base AI provider abstraction
│   ├── gemini.py          # Google Gemini integration
│   └── openrouter.py      # OpenRouter integration
├── attestation/           # TEE attestation implementation
│   ├── __init__.py        # Package exports
│   ├── simulated_token.txt # Testing token
│   ├── vtpm_attestation.py # vTPM client for secure attestation
│   └── vtpm_validation.py  # Token validation utilities
├── prompts/               # Prompt engineering templates
│   ├── __init__.py        # Exports prompt constants
│   └── templates.py       # Different prompt strategies
├── __init__.py            # Package initialization
├── main.py                # Application entry point
├── settings.py            # Configuration settings
└── tune_model.py          # Model fine-tuning utilities
```

### Core Scripts and Their Functions

#### `ai/base.py`
Defines the abstraction layer for AI providers with standardized interfaces:
- `BaseAIProvider`: Abstract class that all AI providers must implement
- `ModelResponse`: Standard response format across different AI backends
- `BaseRouter`/`AsyncBaseRouter`: HTTP request handlers for API communication
- Supports both synchronous and asynchronous request patterns

#### `ai/gemini.py`
Google Gemini implementation of the AI provider interface:
- Handles authentication with Google's Generative AI service
- Manages chat history and context
- Supports structured outputs and system instructions
- Implements conversation memory for multi-turn interactions

#### `attestation/vtpm_attestation.py`
TEE-based security implementation:
- `Vtpm` class for interacting with Trusted Execution Environment
- Generates attestation tokens for verifying secure execution
- Supports nonce-based replay protection
- Can operate in simulation mode for testing and development

#### `attestation/vtpm_validation.py`
Verification utilities for attestation tokens:
- Validates certificate chains and signatures
- Supports both PKI and OIDC validation schemes
- Verifies hardware integrity and runtime environment
- Ensures temporal validity of certificates

#### `prompts/templates.py`
Collection of prompt engineering strategies:
- `ZERO_SHOT_PROMPT`: Basic personality definition for the AI agent
- `FEW_SHOT_PROMPT`: Enhanced with examples for better alignment with communication style
- `CHAIN_OF_THOUGHT_PROMPT`: Detailed reasoning framework for complex queries
- Structured reasoning patterns for different query types

#### `settings.py`
Configuration management using Pydantic:
- API credentials for AI providers
- Model tuning parameters (batch size, learning rate, epochs)
- Dataset paths and model identifiers
- Social media platform credentials (X/Twitter API integration)
- Environment variables management with `.env` file support

#### `tune_model.py`
Utilities for fine-tuning models:
- Data loading and validation with minimum dataset size checks
- Model creation, training, and management
- Training visualization and loss metrics analysis
- Performance monitoring with snapshot tracking
- Model deletion and recreation capabilities

#### `main.py`
Application entry point that demonstrates:
- Testing different prompt strategies against standard prompts
- Loading and using tuned models from Google AI studio
- Comparing response quality across different prompting approaches
- Structured logging for result analysis
- Planned functionality for social media integration

### Using the Code

The template provides core functionality for:

1. **Model Tuning**: Fine-tune AI models on custom datasets
   ```python
   # From tune_model.py
   operation = genai.create_tuned_model(
       id=new_model_id,
       source_model=settings.tuning_source_model,
       training_data=training_dataset,
       epoch_count=settings.tuning_epoch_count,
       batch_size=settings.tuning_batch_size,
       learning_rate=settings.tuning_learning_rate,
   )
   
   # Monitor training progress
   for _ in operation.wait_bar():
       pass
       
   # Analyze and visualize results
   snapshots = pd.DataFrame(tuned_model.tuning_task.snapshots)
   plot_path = save_loss_plot(snapshots, new_model_id)
   ```

2. **AI Interaction with Different Prompting Strategies**:
   ```python
   # From main.py - Zero shot prompting
   model_zero_shot = GeminiProvider(
       settings.gemini_api_key,
       model="gemini-1.5-flash",
       system_instruction=ZERO_SHOT_PROMPT,
   )
   
   # Few shot prompting with examples
   model_few_shot = GeminiProvider(
       settings.gemini_api_key,
       model="gemini-1.5-flash",
       system_instruction=FEW_SHOT_PROMPT,
   )
   
   # Chain of thought reasoning
   model_chain_of_thought = GeminiProvider(
       settings.gemini_api_key,
       model="gemini-1.5-flash",
       system_instruction=CHAIN_OF_THOUGHT_PROMPT,
   )
   ```

3. **Response Testing and Analysis**:
   ```python
   # From main.py
   def test_prompts(model: GeminiProvider, label: str) -> None:
       for prompt in TEST_PROMPTS:
           result = model.generate(prompt)
           logger.info(label, prompt=prompt, result=result.text)
   
   # Test with standard prompts
   test_prompts(model=model_zero_shot, label="zero-shot")
   test_prompts(model=model_few_shot, label="few-shot")
   test_prompts(model=model_chain_of_thought, label="chain-of-thought")
   ```

4. **Secure Attestation** (when implementing TEE security):
   ```python
   # Example attestation usage
   from flare_ai_social.attestation import Vtpm, VtpmValidation
   
   # Generate attestation token
   vtpm = Vtpm()
   token = vtpm.get_token(nonces=["random_nonce"])
   
   # Validate token
   validator = VtpmValidation()
   claims = validator.validate_token(token)
   ```

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
   An example dataset is provided in `src/data/training_data.json`, which consists of tweets from
   [Hugo Philion's X](https://x.com/HugoPhilion) account. You can use any publicly available dataset
   for model fine-tuning.

3. **Tune a new model:**
   Set the name of the new tuned model in `src/flare_ai_social/tune_model.py`, then:

   ```bash
   uv run start-tuning
   ```

4. **Observe loss parameters:**
   After tuning is complete, a training loss PNG will be saved in the root folder corresponding to the new model.
   Ideally, the loss should minimize to near 0 after several training epochs.

5. **Test the new model:**
   Select the new tuned model and test it against a set of prompts:

   ```bash
   uv run start-social
   ```