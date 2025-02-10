from github import Github, GithubException
import os
import base64
import logging

# Настройка логирования
logger = logging.getLogger(__name__)

# GitHub repository settings (can be overridden by environment variables)
GITHUB_USERNAME = os.environ.get("GITHUB_USERNAME") or "Azerus96"
GITHUB_REPOSITORY = os.environ.get("GITHUB_REPOSITORY") or "gemiprofc"
AI_PROGRESS_FILENAME = "cfr_data.pkl"

def save_ai_progress_to_github(filename=AI_PROGRESS_FILENAME):
    token = os.environ.get("AI_PROGRESS_TOKEN")
    if not token:
        logger.warning("AI_PROGRESS_TOKEN not set. Progress saving to GitHub disabled.")
        return False # Indicate failure

    try:
        g = Github(token)
        repo = g.get_user(GITHUB_USERNAME).get_repo(GITHUB_REPOSITORY)

        try:
            contents = repo.get_contents(filename, ref="main")
            with open(filename, 'rb') as f:
                content = f.read()
            repo.update_file(contents.path, "Update AI progress", base64.b64encode(content).decode('utf-8'), contents.sha, branch="main")
            logger.info(f"AI progress saved to GitHub: {GITHUB_REPOSITORY}/{filename}")
            return True # Indicate success
        except GithubException as e:
            if e.status == 404:
                with open(filename, 'rb') as f:
                    content = f.read()
                repo.create_file(filename, "Initial AI progress", base64.b64encode(content).decode('utf-8'), branch="main")
                logger.info(f"Created new file for AI progress on GitHub: {GITHUB_REPOSITORY}/{filename}")
                return True # Indicate success
            else:
                logger.error(f"Error saving progress to GitHub (other than 404): {e}")
                return False # Indicate failure

    except GithubException as e:
        logger.error(f"Error saving progress to GitHub: {e}")
        return False # Indicate failure
    except Exception as e:
        logger.exception(f"An unexpected error occurred during saving: {e}")  # Лучше использовать logger.exception
        return False # Indicate failure


def load_ai_progress_from_github(filename=AI_PROGRESS_FILENAME):
    token = os.environ.get("AI_PROGRESS_TOKEN")
    if not token:
        logger.warning("AI_PROGRESS_TOKEN not set. Progress loading from GitHub disabled.")
        return False # Indicate failure

    try:
        g = Github(token)
        repo = g.get_user(GITHUB_USERNAME).get_repo(GITHUB_REPOSITORY)
        contents = repo.get_contents(filename, ref="main")
        file_content = base64.b64decode(contents.content)
        with open(filename, 'wb') as f:
            f.write(file_content)
        logger.info(f"AI progress loaded from GitHub: {GITHUB_REPOSITORY}/{filename}")
        return True # Indicate success

    except GithubException as e:
        if e.status == 404:
            logger.info("Progress file not found in GitHub repository.")
            return False # Indicate failure
        else:
            logger.error(f"Error loading progress from GitHub: {e}")
            return False # Indicate failure
    except Exception as e:
        logger.exception(f"An unexpected error occurred during loading: {e}")  # Лучше использовать logger.exception
        return False # Indicate failure
