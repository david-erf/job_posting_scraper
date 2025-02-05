import openai
import os 
import pandas as pd

client = openai.OpenAI()

openai.api_key = os.getenv("OPENAI_API_KEY")


resume="  DAVID B. ERF (847) 431-3373 • davidberf@gmail.com EXPERIENCE Pinnacle Healthcare Advisors San Francisco, CA Data Scientist Nov 2022 – Present • Design and implement data pipelines to replace outgoing vendor, producing $246K in annual savings; translate business logic into highly configurable JupyterLab notebooks to automate FTE work distribution • Build and manage a database to track and prioritize appx. 225K claims monthly (totaling $148M in revenue); identified erroneous data processing thereby recovering $450K in ongoing revenue monthly Avant (Series E Online Lender with $9B in Loans Funded) Chicago, IL Manager (Product Analytics), Auto Loans and Unsecured Installment Loans May 2020 – Sept 2022 • Managed team of analysts to make clear, data-driven proposals directly to Executives and Governance committees • Lead A/B testing of improved XGB credit model for flagship product, recommending 12% volume increase • Maintained git-based SQL/Pandas framework to validate product attributes, reducing review time from 45 minutes to less than a minute; design/build portfolio monitoring assets and dashboards via AWS data warehouse • Credit policy owner for ‘Refi+’, Avant’s first secured lending product launched publicly in Q3 2021; $11M funded Manager (Product Analytics), Credit Cards May 2019 – Apr 2020 • Built initial version ML model suite in scikit-learn to identify high-risk customers through early spending, enabling 30% increase in card originations and 15% reduction in projected write-offs; extension considered to combat fraud • Earned firm’s “Get Sh!t Done” award for coordinating with business ops, data engineering/science, and legal teams • Deployed models by joining internal tools (Hadoop, R, and Github) to legacy banking systems, saving 600 employee- hours annually and eliminating $200k in external service costs • Interviewed fraud analysts to turn ad hoc queries into formal account monitoring tools in Jupyter, reducing loan application review times by 50% and creating audit trails for compliance Software Product Manager, Personal Unsecured Loans Jun 2017 – Apr 2019 • Oversaw firm’s customer onboarding website and CRM software, servicing 5,000 applicants and 200 employees daily • Directed a team of 7 engineers and 2 analysts to adapt customer verification system for sale as a SaaS product (spun off as part of Amount in Q4 2019), onboarding 3 new bank partnerships; led daily check-ins with key stakeholders to decisively prioritize requests and translate business needs into technical specs • Supervised A/B testing framework for multi-tenancy capabilities while enhancing funnel performance metrics • Collaborated with data science to improve statistical model for customer income prediction, reducing high-friction customer touches by 10% and enabling 5% increase in conversion Senior Associate and ETL Engineer, Data Services Aug 2015 – Jun 2017 • Implemented Hadoop Data Warehouse on team of 5 engineers; led weekly cross-departmental meetings to test various SQL engines (Snowflake, Hive, Spark SQL and Presto), allowing seamless transition • Managed team of 3 business intelligence analysts to develop original reporting for Treasury, Accounting, and Financial Planning teams, earning firm’s “Get Sh!t Done” award for ensuring ongoing funding from initial investors Analysis Group (Economic, Financial, and Strategy Consulting Firm) Los Angeles, CA Senior Analyst, promoted from Analyst Aug 2013 – Aug 2015 • Led team of 3 analysts in 4.5 year healthcare analytics project for nation's largest nonprofit healthcare provider EDUCATION University of California, Berkeley Master of Information and Data Science (Part-Time) Pomona College Bachelor of Arts in Mathematics (GPA: 3.79/4.00) SKILLS, INTERESTS Berkeley, CA 2025 (Anticipated) Claremont, CA May 2013 Skilled in SQL, Python (pandas, matplot, seaborn, and sckit-learn among others), Jupyter Lab, and Excel/VBA • Competitive programming; woodworking; gardening; running; indoor rock climbing"

ex_job_posting="SoFi provided pay rangelearn more.Base pay range$18.00/hr - $20.00/hrEmployee Applicant Privacy NoticeWho we are:Shape a brighter financial future with us.about and interact with personal finance.yourself, your career, and the financial world.The roleMember’s concerns and seeing them through to resolution.What you’ll do:First Call Resolution (FCR)Samsung Money by SoFi, and SoFi Invest productsnotating correspondence after each customer contact is handledforesee causes for additional inquiriesvarious tools/features to help them get their money rightconversations to deescalate simple or complex inquiriesthe core responsibilities of the roleoperations, and policiesbusiness based on business needssend emails, and perform callbacks to client customersconcernsthe best solutions to ensure customer requirements are metfunctional areas and with external clients.Member Experience standardsWhat you’ll need:industry or Call Center environmentStrong verbal and written communication skillsenvironmentprovide effective solutionsBasic computer skills with solid proficiency in Google Suiteconcernswith professionalismproducts, and servicesrequestsinquiries simultaneously-6:00 PM MSTchannels of communicationHigh school diploma or GED requiredMust successfully pass FINRA fingerprint background checktraining may be in officeAbility to work 2 days in office after training is completedJacksonville, Florida location onlyHourly pay $19.50Compensation And Benefitsas the candidate’s experience, skills, and location.visit our Benefits at SoFi page!prohibited by applicable state or federal law.without regard to protected characteristics.conviction records.New York applicants: Notice of Employee Rightsaccommodations@sofi.com.remote work from Hawaii or Alaska at this time.Internal Employeesour open roles."

def rank_job_posting(resume, job_posting):
    """
    Ranks a list of job postings based on their relevance to the given resume.
    
    Parameters:
    - resume (str): The candidate's resume as a string.
    - job_posting (str): description or title as a string. 
    
    Returns:
    - List of tuples (job_posting, score) sorted by relevance score in descending order.
    """
    prompt = f"""
    You are an AI job relevance scorer. Given a resume and a job posting, rate how relevant the job is for the candidate on a scale from 1 to 10 (10 being the most relevant).
    
    ### Resume:
    {resume}
    
    ### Job Posting:
    {job_posting}
    
    Provide only the numeric score.
    """

    try:
        response = openai.chat.completions.create(
            model="gpt-4-turbo",
            messages=[{"role": "user", "content": prompt}]
        )
        # Extract and convert score to an integer
        score = int(response.choices[0].message.content.strip())
        ans = score
    except Exception as e:
        print(f"Error processing job posting: {job_posting}\nError: {e}")
        ans=0

    print(job_posting[0:50],ans)
    
    return ans




