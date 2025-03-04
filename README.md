# Flare AI Social

Flare AI Kit template for Social AI Agents.

## 🚀 Key Features

- **Secure AI Execution**  
  Runs within a Trusted Execution Environment (TEE) featuring remote attestation support for robust security.

- **Built-in Chat UI**  
  Interact with your AI via a TEE-served chat interface.

- **Gemini Fine-Tuning Support**  
  Fine-tune foundational models with custom datasets.

- **Social media integrations**  
  X and Telegram integrations with with rate limiting and retry mechanisms.

## 🎯 Getting Started

### Prerequisites

- [uv](https://docs.astral.sh/uv/getting-started/installation/)
- [Docker](https://www.docker.com/)

### Fine-tune a model

1. **Prepare Environment File**: Rename `.env.example` to `.env` and update these model fine-tuning parameters:

   | Parameter             | Description                                                               | Default                            |
   | --------------------- | ------------------------------------------------------------------------- | ---------------------------------- |
   | `tuned_model_name`    | Name of the newly tuned model                                             | pugo-hilion                        |
   | `tuning_source_model` | Name of the foundational model to tune on                                 | models/gemini-1.5-flash-001-tuning |
   | `epoch_count`         | Number of tuning epochs to run. An epoch is a pass over the whole dataset | 30                                 |
   | `batch_size`          | Number of examples to use in each training batch                          | 4                                  |
   | `learning_rate`       | Step size multiplier for the gradient updates                             | 0.001                              |

   Ensure your [Gemini API key](https://aistudio.google.com/app/apikey) is setup.

2. **Install dependencies:**

   ```bash
   uv sync --all-extras
   ```

3. **Prepare a dataset:**
   An example dataset is provided in `src/data/training_data.json`, which consists of tweets from
   [Hugo Philion's X](https://x.com/HugoPhilion) account. You can use any publicly available dataset
   for model fine-tuning.

4. **Tune a new model:**
   Depending on the size of your dataset, this process can take several minutes:

   ```bash
   uv run start-tuning
   ```

5. **Observe loss parameters:**
   After tuning in complete, a training loss PNG will be saved in the root folder corresponding to the new model.
   Ideally the loss should minimize to near 0 after several training epochs.

   ![agent-pugo-hilion_mean_loss](https://github.com/user-attachments/assets/39882da7-8f5f-45cd-afca-709f1333edf4)

6. **Test the new model:**
   Select the new tuned model and compare it against a set of prompting techniques (zero-shot, few-shot and chain-of-thought):

   ```bash
   uv run start-compare
   ```

7. **Start Social Bots (optional):**

   - Set up Twitter/X API credentials
   - Configure Telegram bot token
   - Enable/disable platforms as needed

   ```bash
   uv run start-bots
   ```

### Build Using Docker (Recommended)

The Docker setup mimics a TEE environment and includes an Nginx server for routing, while Supervisor manages both the backend and frontend services in a single container.

1. **Build the Docker image**:

   ```bash
   docker build -t flare-ai-social .
   ```

   **NOTE:** Windows users may encounter issues with `uv` due to incorrect parsing. For this try converting the `pyproject.toml` and `uv.lock` files to unix format.

2. **Run the Docker Container:**

   ```bash
   docker run -p 80:80 -it --env-file .env flare-ai-social
   ```

3. **Access the Frontend:**  
   Open your browser and navigate to [http://localhost:80](http://localhost:80) to interact with the tuned model via the Chat UI.

### 🛠 Build Manually

Flare AI Social is composed of a Python-based backend and a JavaScript frontend. Follow these steps for manual setup:

#### Backend Setup

1. **Install Dependencies:**
   Use [uv](https://docs.astral.sh/uv/getting-started/installation/) to install backend dependencies:

   ```bash
   uv sync --all-extras
   ```

2. **Start the Backend:**
   The backend runs by default on `0.0.0.0:80`:

   ```bash
   uv run start-backend
   ```

#### Frontend Setup

1. **Install Dependencies:**
   In the `chat-ui/` directory, install the required packages using [npm](https://nodejs.org/en/download):

   ```bash
   cd chat-ui/
   npm install
   ```

2. **Configure the Frontend:**
   Update the backend URL in `chat-ui/src/components/ChatInterface.js` for testing:

   ```js
   const BACKEND_ROUTE = "http://localhost:8080/api/routes/chat/";
   ```

   > **Note:** Remember to change `BACKEND_ROUTE` back to `'api/routes/chat/'` after testing.

3. **Start the Frontend:**

   ```bash
   npm start
   ```

## 📁 Repo Structure

```plaintext
src/flare_ai_social/
├── ai/                            # AI Provider implementations
│   ├── base.py                    # Base AI provider abstraction
│   ├── gemini.py                  # Google Gemini integration
│   └── openrouter.py             # OpenRouter integration
├── api/                           # API layer
│   └── routes/                    # API endpoint definitions
├── attestation/                   # TEE attestation implementation
│   ├── vtpm_attestation.py       # vTPM client
│   └── vtpm_validation.py        # Token validation
├── prompts/                       # Prompt engineering templates
│   └── templates.py              # Different prompt strategies
├── telegram/                      # Telegram bot implementation
│   └── service.py                # Telegram service logic
├── twitter/                       # Twitter bot implementation
│   └── service.py                # Twitter service logic
├── bot_manager.py                # Bot orchestration
├── main.py                       # FastAPI application
├── settings.py                   # Configuration settings
└── tune_model.py                 # Model fine-tuning utilities
```

## 🚀 Deploy on TEE

Deploy on [Confidential Space](https://cloud.google.com/confidential-computing/confidential-space/docs/confidential-space-overview) using AMD SEV.

### Prerequisites

- **Google Cloud Platform Account:**  
  Access to the [`verifiable-ai-hackathon`](https://console.cloud.google.com/welcome?project=verifiable-ai-hackathon) project is required.

- **Gemini API Key:**  
  Ensure your [Gemini API key](https://aistudio.google.com/app/apikey) is linked to the project.

- **gcloud CLI:**  
  Install and authenticate the [gcloud CLI](https://cloud.google.com/sdk/docs/install).

### Environment Configuration

1. **Set Environment Variables:**  
   Update your `.env` file with:

   ```bash
   TEE_IMAGE_REFERENCE=ghcr.io/YOUR_REPO_IMAGE:main  # Replace with your repo build image
   INSTANCE_NAME=<PROJECT_NAME-TEAM_NAME>
   ```

2. **Load Environment Variables:**

   ```bash
   source .env
   ```

   > **Reminder:** Run the above command in every new shell session or after modifying `.env`. On Windows, we recommend using [git BASH](https://gitforwindows.org) to access commands like `source`.

3. **Verify the Setup:**

   ```bash
   echo $TEE_IMAGE_REFERENCE # Expected output: Your repo build image
   ```

### Deploying to Confidential Space

Run the following command:

```bash
gcloud compute instances create $INSTANCE_NAME \
  --project=verifiable-ai-hackathon \
  --zone=us-central1-c \
  --machine-type=n2d-standard-2 \
  --network-interface=network-tier=PREMIUM,nic-type=GVNIC,stack-type=IPV4_ONLY,subnet=default \
  --metadata=tee-image-reference=$TEE_IMAGE_REFERENCE,\
tee-container-log-redirect=true,\
tee-env-GEMINI_API_KEY=$GEMINI_API_KEY,\
tee-env-TUNED_MODEL_NAME=$TUNED_MODEL_NAME,\
  --maintenance-policy=MIGRATE \
  --provisioning-model=STANDARD \
  --service-account=confidential-sa@verifiable-ai-hackathon.iam.gserviceaccount.com \
  --scopes=https://www.googleapis.com/auth/cloud-platform \
  --min-cpu-platform="AMD Milan" \
  --tags=flare-ai,http-server,https-server \
  --create-disk=auto-delete=yes,\
boot=yes,\
device-name=$INSTANCE_NAME,\
image=projects/confidential-space-images/global/images/confidential-space-debug-250100,\
mode=rw,\
size=11,\
type=pd-standard \
  --shielded-secure-boot \
  --shielded-vtpm \
  --shielded-integrity-monitoring \
  --reservation-affinity=any \
  --confidential-compute-type=SEV
```

#### Post-deployment

1. After deployment, you should see an output similar to:

   ```plaintext
   NAME          ZONE           MACHINE_TYPE    PREEMPTIBLE  INTERNAL_IP  EXTERNAL_IP    STATUS
   social-team1   us-central1-a  n2d-standard-2               10.128.0.18  34.41.127.200  RUNNING
   ```

2. It may take a few minutes for Confidential Space to complete startup checks. You can monitor progress via the [GCP Console](https://console.cloud.google.com/welcome?project=verifiable-ai-hackathon) logs.
   Click on **Compute Engine** → **VM Instances** (in the sidebar) → **Select your instance** → **Serial port 1 (console)**.

   When you see a message like:

   ```plaintext
   INFO:     Uvicorn running on http://0.0.0.0:8080 (Press CTRL+C to quit)
   ```

   the container is ready. Navigate to the external IP of the instance (visible in the **VM Instances** page) to access the Chat UI.

### 🔧 Troubleshooting

If you encounter issues, follow these steps:

1. **Check Logs:**

   ```bash
   gcloud compute instances get-serial-port-output $INSTANCE_NAME --project=verifiable-ai-hackathon
   ```

2. **Verify API Key(s):**  
   Ensure that all API Keys are set correctly (e.g. `GEMINI_API_KEY`).

3. **Check Firewall Settings:**  
   Confirm that your instance is publicly accessible on port `80`.

## 💡 Next Steps

Below are several project ideas demonstrating how the template can be used to build useful social AI agents:

### Dev Support on Telegram

- **Integrate with flare-ai-rag:**  
  Combine the social AI agent with the [flare-ai-rag](https://github.com/flare-foundation/flare-ai-rag) model trained on the [Flare Developer Hub](https://dev.flare.network) dataset.
- **Enhanced Developer Interaction:**

  - Provide targeted support for developers exploring [FTSO](https://dev.flare.network/ftso/overview) and [FDC](https://dev.flare.network/fdc/overview).
  - Implement code-based interactions, including live debugging tips and code snippet sharing.

- **Action Steps:**
  - Connect the model to GitHub repositories to fetch live code examples.
  - Fine-tune prompt templates using technical documentation to improve precision in code-related queries.

### Community Support on Telegram

- **Simplify Technical Updates:**
  - Convert detailed [Flare governance proposals](https://proposals.flare.network) into concise, accessible summaries for community members.
- **Real-Time Monitoring and Q&A:**

  - Monitor channels like the [Flare Telegram](https://t.me/FlareNetwork) for live updates.
  - Automatically answer common community questions regarding platform changes.

- **Action Steps:**
  - Integrate modules for content summarization and sentiment analysis.
  - Establish a feedback loop to refine responses based on community engagement.

### Social Media Sentiment & Moderation Bot

- **Purpose:**  
  Analyze sentiment on platforms like Twitter, Reddit, or Discord to monitor community mood, flag problematic content, and generate real-time moderation reports.

- **Action Steps:**
  - Leverage NLP libraries for sentiment analysis and content filtering.
  - Integrate with social media APIs to capture and process live data.
  - Set up dashboards to monitor trends and flagged content.

### Personalized Content Curation Agent

- **Purpose:**  
  Curate personalized content such as news, blog posts, or tutorials tailored to user interests and engagement history.

- **Action Steps:**
  - Employ user profiling techniques to analyze preferences.
  - Use machine learning algorithms to recommend content based on past interactions.
  - Continuously refine the recommendation engine with user feedback and engagement metrics.
