# Security reporting

Do not disclose API keys, participant recordings, or device access details in
public issues. Use GitHub's private vulnerability reporting when it is available
on this repository; otherwise use the contact channel on
[Jizai's website](https://jizai.ai) to request a private reporting route before
sharing sensitive details.

This sample uses your OpenAI project key on the trusted robot host. The app's
license does not grant access to OpenAI services. Keep credentials out of source
control and review terminal/service logs before sharing them.

The application is intended for supervised hardware operation. Spoken stop is
not a physical emergency-stop mechanism. A software patch must pass dry-run
tests and supervised hardware checks before relying on changed motion behavior.
