# PDF → LLM → Slack → Google Calendar

A Python automation workflow that:

1. Extracts text, metadata, headings, and tables from one or more PDFs.
2. Sends the extracted content to OpenAI, Anthropic Claude, or Google Gemini and
   requests a structured summary with insights, takeaways, action items,
   important dates, and a proposed follow-up meeting.
3. Sends a formatted Block Kit summary to Slack through an incoming webhook or
   the Slack Web API.
4. Creates a Google Calendar follow-up event using either a service account or
   OAuth 2.0.
5. Writes a local JSON and Markdown report for auditing.

The Google Calendar event is scheduled before the Slack message is sent, so the
Slack message can include the calendar link.

## Requirements

- Python 3.10+
- An API key for one supported LLM provider
- A Slack incoming webhook or bot token
- A Google Cloud service account or OAuth client for Google Calendar

## Project structure

```text
AIbot/
├── run.py                     # Convenience entry point
├── requirements.txt
├── .env.example
├── pdf_workflow/
│   ├── cli.py                 # Argument parsing and user-facing output
│   ├── config.py              # .env loading and validation
│   ├── models.py              # Structured LLM output models
│   ├── pdf_processor.py       # PDF text/structure extraction
│   ├── llm_client.py          # OpenAI / Anthropic / Gemini adapters
│   ├── slack_client.py        # Slack webhook/bot delivery and formatting
│   ├── calendar_client.py     # Google Calendar auth and event creation
│   ├── reporting.py           # Local JSON/Markdown reports
│   ├── workflow.py            # End-to-end orchestration
│   ├── __init__.py            # Package version
│   └── __main__.py            # `python -m pdf_workflow` entry point
├── local_english_model/       # Separate offline GPT-style text model
└── outputs/                   # Created automatically at runtime
```

## Setup

```bash
cd AIbot
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
mkdir -p credentials
```

Edit `.env` and fill in the provider, Slack, and Google Calendar values.

Do not commit `.env` or credential files. `.gitignore` already excludes them.

## LLM configuration

Set one provider and its key:

```dotenv
LLM_PROVIDER=openai
OPENAI_API_KEY=...
# LLM_MODEL=gpt-4.1-mini
```

Supported providers:

| `LLM_PROVIDER` | Required key | Default model |
|---|---|---|
| `openai` | `OPENAI_API_KEY` | `gpt-4.1-mini` |
| `anthropic` | `ANTHROPIC_API_KEY` | `claude-sonnet-4-5` |
| `gemini` | `GEMINI_API_KEY` or `GOOGLE_API_KEY` | `gemini-2.5-flash` |

Model names change over time. Set `LLM_MODEL` to a model available to your
account if the default is unavailable. The LLM is asked to return strict JSON;
if the first response cannot be parsed, the workflow makes one repair request.
Transient LLM failures are retried with exponential backoff.

## Slack setup

### Option A: incoming webhook (simplest)

1. Go to <https://api.slack.com/apps> and create an app.
2. Open **Incoming Webhooks**, activate it, and add a webhook to the target
   channel.
3. Copy the webhook URL into `.env`:

```dotenv
SLACK_ENABLED=true
SLACK_MODE=webhook
SLACK_WEBHOOK_URL=https://hooks.slack.com/services/...
```

### Option B: Slack Web API bot

1. Create a Slack app and add the `chat:write` bot scope.
2. Install the app to the workspace.
3. Invite the bot to the target channel (`/invite @your-bot`).
4. Copy the bot token and channel ID:

```dotenv
SLACK_ENABLED=true
SLACK_MODE=bot
SLACK_BOT_TOKEN=xoxb-...
SLACK_CHANNEL_ID=C0123456789
```

`SLACK_MODE=auto` (the default) uses the webhook when `SLACK_WEBHOOK_URL` is set,
otherwise it uses the bot token and channel.

## Google Calendar setup

### Option A: service account (recommended for unattended automation)

1. In the Google Cloud Console, create or select a project.
2. Enable the **Google Calendar API**.
3. Create a service account and download a JSON key.
4. Save it as `credentials/google-service-account.json`.
5. Copy the service account email address.
6. In Google Calendar, open the target calendar's settings and share it with the
   service account email, granting **Make changes to events**.
7. Set the calendar ID in `.env`. For a shared calendar this is usually the
   calendar owner's email or the ID from **Calendar settings → Integrate
   calendar**. `primary` refers to the service account's own calendar, which is
   usually not what you want for service accounts.

```dotenv
CALENDAR_ENABLED=true
GOOGLE_AUTH_MODE=service_account
GOOGLE_SERVICE_ACCOUNT_FILE=./credentials/google-service-account.json
GOOGLE_CALENDAR_ID=owner@example.com
TIMEZONE=America/New_York
```

### Option B: OAuth 2.0

1. Configure an OAuth consent screen in Google Cloud and add yourself as a test
   user if the app is still in testing.
2. Create an OAuth client ID of type **Desktop app** and download the JSON.
3. Configure:

```dotenv
CALENDAR_ENABLED=true
GOOGLE_AUTH_MODE=oauth
GOOGLE_OAUTH_CLIENT_SECRET_FILE=./credentials/google-oauth-client.json
GOOGLE_OAUTH_TOKEN_FILE=./credentials/google-token.json
GOOGLE_CALENDAR_ID=primary
TIMEZONE=America/New_York
```

The first run opens a browser for authorization and saves the token to
`GOOGLE_OAUTH_TOKEN_FILE` with mode `0600`. OAuth cannot complete a browser flow
on a headless server unless you perform the authorization on a machine with a
browser and copy the token file, or use a service account instead.

### Follow-up scheduling

The LLM proposes a follow-up date/time from document deadlines, discussion
topics, decisions, and action items. The scheduler then:

- Uses the suggested date/time when possible.
- Falls back to `FOLLOWUP_DEFAULT_DAYS` business days ahead at
  `FOLLOWUP_DEFAULT_TIME`.
- Moves past or invalid dates to the next business day.
- Uses `FOLLOWUP_DURATION_MINUTES` unless the LLM supplied a duration.
- Invites only addresses in `FOLLOWUP_ATTENDEES`; names from the document are
  included in the description but no email addresses are invented.
- Reuses an equivalent event when `CALENDAR_DEDUPE=true` and the same source
  fingerprint/time already exists.

Set `FOLLOWUP_ALWAYS_SCHEDULE=true` to schedule even when the LLM says a meeting
is not required.

## Run

```bash
# Normal run
python run.py /path/to/document.pdf

# Multiple PDFs are analyzed as one bundle with one summary and one follow-up
python run.py board-pack.pdf appendix.pdf

# Validate extraction, LLM output, and message formatting without sending
python run.py document.pdf --dry-run

# Skip an integration for this run
python run.py document.pdf --skip-calendar
python run.py document.pdf --skip-slack

# Force a follow-up even if the LLM says one is not needed
python run.py document.pdf --force-followup

# Custom env file, output directory, or no local report
python run.py document.pdf --env-file .env.production --output-dir reports
python run.py document.pdf --no-report
```

The package can also be run as a module:

```bash
python -m pdf_workflow document.pdf
```

`--dry-run` still calls the LLM, but it does not send a Slack message or create
a calendar event. It prints the formatted Slack preview and proposed meeting.

## Local Transformer text model

The repository also contains `local_english_model/`, a separate GPT-style
Transformer trained from scratch on local `.txt` files. It does not use the
`.env` file, an AI provider, or an API key.

```bash
cd local_english_model
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Add your .txt files under local_english_model/data/, then:
python3 -m local_english_model train
python3 -m local_english_model generate "The sun rises in the"
python3 -m local_english_model chat
```

This is a small model, not an instruction-tuned assistant. See
`local_english_model/README.md` for architecture, training options, and
limitations.

## Environment variables

| Variable | Default | Purpose |
|---|---|---|
| `LLM_PROVIDER` | `openai` | `openai`, `anthropic`, or `gemini` |
| `LLM_MODEL` | provider default | Model name |
| `OPENAI_API_KEY` | — | OpenAI API key |
| `ANTHROPIC_API_KEY` | — | Anthropic API key |
| `GEMINI_API_KEY` / `GOOGLE_API_KEY` | — | Gemini API key |
| `LLM_TEMPERATURE` | `0.2` | Sampling temperature |
| `LLM_MAX_OUTPUT_TOKENS` | `4096` | Maximum generated tokens |
| `LLM_TIMEOUT_SECONDS` | `90` | Provider request timeout |
| `LLM_MAX_ATTEMPTS` | `3` | Total attempts for transient LLM errors |
| `PDF_MAX_CHARS` | `80000` | Maximum characters sent to the LLM |
| `PDF_MAX_PAGES` | `0` | Process only the first N pages; `0` = all |
| `SLACK_ENABLED` | `true` | Enable Slack notifications |
| `SLACK_MODE` | `auto` | `auto`, `webhook`, or `bot` |
| `SLACK_WEBHOOK_URL` | — | Incoming webhook URL |
| `SLACK_BOT_TOKEN` | — | Bot token with `chat:write` |
| `SLACK_CHANNEL_ID` | — | Channel ID or `#channel-name` |
| `SLACK_TIMEOUT_SECONDS` | `20` | Slack Web API timeout |
| `CALENDAR_ENABLED` | `true` | Enable Calendar scheduling |
| `GOOGLE_AUTH_MODE` | `service_account` | `service_account` or `oauth` |
| `GOOGLE_SERVICE_ACCOUNT_FILE` | — | Service account JSON path |
| `GOOGLE_SERVICE_ACCOUNT_SUBJECT` | — | Optional delegation subject |
| `GOOGLE_OAUTH_CLIENT_SECRET_FILE` | — | OAuth desktop client JSON path |
| `GOOGLE_OAUTH_TOKEN_FILE` | `./credentials/google-token.json` | Saved OAuth token |
| `GOOGLE_CALENDAR_ID` | `primary` | Target Google Calendar ID |
| `TIMEZONE` | `UTC` | IANA timezone for scheduling |
| `FOLLOWUP_DEFAULT_DAYS` | `3` | Fallback business days ahead |
| `FOLLOWUP_DEFAULT_TIME` | `10:00` | Fallback local start time |
| `FOLLOWUP_DURATION_MINUTES` | `30` | Fallback meeting duration |
| `FOLLOWUP_ATTENDEES` | empty | Comma-separated invitee emails |
| `FOLLOWUP_ALWAYS_SCHEDULE` | `false` | Always create a follow-up |
| `CALENDAR_DEDUPE` | `true` | Reuse equivalent existing events |
| `OUTPUT_DIR` | `outputs` | Local report directory |
| `LOG_LEVEL` | `INFO` | Python log level |

## Output and errors

- Each successful run writes `<timestamp>_<title>.json` and `.md` under
  `OUTPUT_DIR` unless `--no-report` is used.
- PDF warnings, LLM summary, scheduling details, Slack delivery metadata, and
  errors are recorded in the reports.
- Slack or Calendar failures do not prevent the other steps from running. The
  process exits with code `1` if any step fails, after printing the collected
  errors.
- PDF, configuration, and LLM failures are fatal and produce a non-zero exit.
- Scanned/image-only PDFs must be OCR-processed before running this workflow.

## Security notes

- Keep `.env`, service account keys, OAuth client secrets, and token files out
  of version control.
- Use a least-privilege Slack bot and share only the required Google Calendar.
- PDF content is sent to the selected LLM provider. Do not process documents
  that are not permitted to leave your environment.
- LLM output can be inaccurate. Review generated summaries, action items, and
  meeting invitations before relying on them.
