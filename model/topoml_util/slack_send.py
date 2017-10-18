import os
from slackclient import SlackClient

slack_token = os.environ.get("SLACK_API_TOKEN")


def notify(timestamp, script_name, message):
    if slack_token:
        sc = SlackClient(slack_token)
        sc.api_call(
          "chat.postMessage",
          channel="#machinelearning",
          text="Session " + timestamp + ' ' + script_name + " completed successfully with: " + str(message))
    else:
        print('No slack notification: no slack API token environment variable "SLACK_API_TOKEN" set.')
