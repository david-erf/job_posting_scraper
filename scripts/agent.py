import subprocess


def call_ollama_run(model: str, prompt: str) -> str:
    """
    Calls the Ollama CLI using the `run` command with a given prompt and returns the model's output.
    
    Parameters:
        model (str): The name of the model to use (e.g., "mistral").
        prompt (str): The prompt to send to the model.
        
    Returns:
        str: The output from the model.
    """
    try:
        result = subprocess.run(
            ["ollama", "run", model],
            input=prompt.encode("utf-8"),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True
        )
        return result.stdout.decode("utf-8").strip()
    except subprocess.CalledProcessError as e:
        error_message = e.stderr.decode("utf-8")
        raise RuntimeError(f"Error calling Ollama: {error_message}")

def rank_job_posting(resume: str, job_posting: str,lim:int = 2000) -> int:
    """
    Uses the Ollama CLI (with the Mistral model) to rank a job posting's relevance
    to the given resume on a scale from 1 to 10.
    
    Parameters:
        resume (str): The candidate's resume.
        job_posting (str): The job posting description or title.
        
    Returns:
        int: The relevance score.
    """
    resume=resume[0:lim]
    job_posting=job_posting[0:lim]
    

    prompt = f"""You are a numeric AI job relevance scorer. For each input, output only a single number (from 1 to 10) on a single line, with no additional text, explanation, or formatting. The number represents the relevance of the job posting to the candidate's resume (10 means highly relevant, 1 means not relevant).

    ### Resume:
    {resume}

    ### Job Posting:
    {job_posting}

    """
    prompt=prompt[0:4000]
    try:
        response = call_ollama_run("mistral", prompt)
        # Parse the response into an integer score
        score = float(response.strip())
    except Exception as e:
        print(f"Error processing job posting: {job_posting}\nError: {e}")
        score = 0

    print(job_posting[:50], score)
    return score