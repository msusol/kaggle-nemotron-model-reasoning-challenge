# Submit to Competition


```
kaggle competitions submit -c nvidia-nemotron-model-reasoning-challenge -f submission.zip -m "Message"
```

## Configuration

```
# Add this to your settings.json:
{
    "servers": {
        "kaggle": {
            "url": "https://www.kaggle.com/mcp",
            "type": "http"
        }
    },
}
```

## Authentication

```
# Call the MCP server's `authorize` tool.

# Or, for token authentication:
{
    "servers": {
        "kaggle": {
            "url": "https://www.kaggle.com/mcp",
            "type": "http",
            "headers" : {
                "authorization": "Bearer <YOUR_TOKEN>"
            }
        }
    },
}
    
# If you don't already have a token, go to Settings > Generate New Token > Copy.
```

## Usage

```
# To upload a submission, prompt your client to use the
# "mcp_kaggle_start_competition_submission_upload" tool. 
# Then, use "kaggle_mcp_submit_to_competition" to submit it to the competition.
```