import streamlit as st

import pandas as pd
import altair as alt


# Load data
df = pd.read_csv('jobs_df_2025-01-13.csv', index_col=False)

df["job_id"] = df["job_id"].astype(str).str.strip().str.lower()

# Streamlit App
st.title("LinkedIn Postings Review")

# Sidebar for page navigation
page = st.sidebar.radio(
    "Choose a view:",
    ("Overall View", "Job Category Split","Job Description Lookup")
)

# Define a function to create the charts
def create_charts(data, filtered=False):
    # Chart 1: Distribution of `max_salary`
    salary_chart = alt.Chart(data).mark_bar().encode(
        x=alt.X(
            "max_salary:Q",
            bin=alt.Bin(step=25000),  # Fixed bin width of 25,000
            scale=alt.Scale(domain=[0, 500000]),  # Fixed axis range to 500k
            axis=alt.Axis(format="$,.0f", tickCount=10, title="Max Salary (Binned)")
        ),
        y=alt.Y("count()", title="Count"),
    ).properties(
        title="Distribution of Max Salary",
        width=400,
        height=300
    )

    # Prepare data for the second chart
    avg_df = data.groupby("time_since_posted_grouping")["gen_ai_flag"].mean().reset_index()

    # Check if any `gen_ai_flag` value exceeds 10%
    max_gen_ai_flag = avg_df["gen_ai_flag"].max()

    # Set dynamic y-axis range: 0-10% as default, expand if necessary
    y_axis_range = [0, max(0.25, max_gen_ai_flag)]
    

    # Chart 2: Combine bar and line charts with shared X-axis
    bar_chart = alt.Chart(data).mark_bar(color="steelblue").encode(
        x=alt.X("time_since_posted_grouping:N", title="Time Since Posted Grouping (Days)"),
        y=alt.Y("count()", title="Count of Grouping")
    )

    line_chart = alt.Chart(avg_df).mark_line(point=True, color="orange").encode(
        x=alt.X("time_since_posted_grouping:N"),
        y=alt.Y(
            "gen_ai_flag:Q", 
            title="Gen AI Adoption Rate",
            scale=alt.Scale(domain=y_axis_range),  # Dynamically adjust range
            axis=alt.Axis(format=".0%")  # Format as percentage
        )
    )

    combined_chart = alt.layer(bar_chart, line_chart).resolve_scale(
        y="independent"  # Keep Y-axes independent
    ).properties(
        title="Counts and Gen AI Adoption Rate by Time Grouping",
        width=400,
        height=300
    )

    # Display charts side-by-side
    st.altair_chart(
        alt.hconcat(salary_chart, combined_chart),
        use_container_width=True
    )

# Page: Overall View
if page == "Overall View":
    st.header("Overview (All Job Postings)")

    # Filtered chart data
    filtered_df = df[(df['max_salary'] < 500000)]
    create_charts(filtered_df)

    # Group data by cat and calculate the required metrics
    category_summary = (
        df.groupby(["job_categorization"])
        .agg(
            total_records=("job_id", "count"),
            total_companies=('searched_company','nunique'),
            avg_min_salary=("min_salary", "mean"),
            avg_max_salary=("max_salary", "mean"),
        )
        .reset_index()
        .sort_values(by='total_records',ascending=False)
    )


    # Group data by company and calculate the required metrics
    company_summary = (
        df.groupby(["searched_company",'company_type'])
        .agg(
            total_records=("job_id", "count"),
            ct_engineering_it = ('jobcat_flag_engineering_it','sum'),
            ct_finance = ('jobcat_flag_finance','sum'),
            ct_hr_admin = ('jobcat_flag_hr_admin','sum'),
            ct_legal = ('jobcat_flag_legal','sum'),
            ct_marketing = ('jobcat_flag_marketing','sum'),
            ct_operations_cs = ('jobcat_flag_operations_cs','sum'),
            ct_product_rnd = ('jobcat_flag_product_rnd','sum'),
            ct_sales = ('jobcat_flag_sales','sum'),            
            avg_min_salary=("min_salary", "mean"),
            avg_max_salary=("max_salary", "mean"),
        )
        .reset_index()
        .sort_values(by='total_records',ascending=False)
    )

    company_summary['company_type'] = company_summary['company_type'].apply(lambda x: 'top50' if x=='none' else x)


    # Display the summary
    st.write("Summary Table by Company:", company_summary)

    # Display the summary
    st.write("Summary Table by Category:", category_summary)

# Page: Job Category Split
elif page == "Job Category Split":
    st.header("Job Category Split")
    
    # Dropdown for user selection
    categories = df["job_categorization"].unique()
    selected_category = st.selectbox("Select a Job Category", categories)

    # Filter dataframe based on selection
    filtered_df = df[(df["job_categorization"] == selected_category) & (df['max_salary'] < 500000)]
    
    create_charts(filtered_df, filtered=True)
    
    display_columns = ['searched_company','job_id','min_salary','max_salary','Seniority', 'Employment',  'Job', 'Industries', 'city','state','job_description',]

    st.write("Filtered Dataframe:", filtered_df[display_columns])

# Page: Job Description Lookup
elif page == "Job Description Lookup":
    st.header("Job Description Lookup")
    
    # Filter job IDs with non-empty job descriptions
    valid_job_ids = df[~df["job_description"].isna() & (df["job_description"] != "")]["job_id"].unique()
    valid_job_ids = sorted(valid_job_ids)  # Sort for better UX
    
    # Dropdown for Job ID selection
    selected_job_id = st.selectbox("Select a Job ID:", valid_job_ids)
    
    if selected_job_id:
        # Retrieve the job description for the selected Job ID
        job_description = df.loc[df["job_id"] == selected_job_id, "job_description"].values[0]
        
        # Display job description as plain text with wrapping
        st.markdown("### Job Description:")
        st.markdown(f"<div style='white-space: pre-wrap; word-wrap: break-word;'>{job_description}</div>", unsafe_allow_html=True)