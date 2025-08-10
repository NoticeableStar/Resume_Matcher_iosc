import numpy as np
import pandas as pd
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
import os

import nltk
nltk.download('punkt_tab')
# Download necessary NLTK data (only first run)
nltk.download('punkt')
nltk.download('stopwords')

# Load SpaCy model 
nlp = spacy.load('en_core_web_sm')

# Load the dataset from local file
data_path = 'Resume.csv'  # Ensure Resume.csv is in the same folder as app.py
df = pd.read_csv(data_path)




# --- Define preprocessing utilities ONCE ---
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def preprocess_text(text):
    tokens = word_tokenize(text)
    tokens = [token for token in tokens if token not in stop_words]
    return ' '.join(tokens)

def lemmatize_text(text):
    doc = nlp(text)
    return ' '.join([token.lemma_ for token in doc if not token.is_punct])

# --- Apply preprocessing pipeline to resumes ---
df['cleaned_resume'] = df['Resume_str'].apply(clean_text)
df['processed_resume'] = df['cleaned_resume'].apply(preprocess_text)
df['lemmatized_resume'] = df['processed_resume'].apply(lemmatize_text)

job_descriptions = {
    "INFORMATION-TECHNOLOGY": """
    Job Title: Senior Software Engineer
    Company: Tech Innovations Inc.
    Location: Bangalore, India
    About Us: We are a fast-growing tech company specializing in AI-driven solutions for the finance sector. We're looking for passionate engineers to join our innovative team.
    Job Description: We are seeking a highly skilled Senior Software Engineer with a strong background in developing robust and scalable applications. You will be responsible for designing, developing, and maintaining software systems, collaborating with cross-functional teams, and contributing to the entire software development lifecycle.
    Responsibilities:
    * Design, develop, and deploy high-quality software solutions.
    * Write clean, maintainable, and efficient code.
    * Participate in code reviews, architectural discussions, and sprint planning.
    * Troubleshoot and debug complex issues.
    * Mentor junior developers and contribute to team growth.
    * Stay updated with emerging technologies and industry best practices.
    Requirements:
    * Bachelor's or Master's degree in Computer Science or a related field.
    * 5+ years of experience in software development.
    * Proficiency in Python, Java, or C++.
    * Experience with cloud platforms (AWS, Azure, GCP).
    * Strong understanding of data structures, algorithms, and distributed systems.
    * Experience with Agile methodologies.
    * Excellent problem-solving and communication skills.
    """,
    "BUSINESS-DEVELOPMENT": """
    Job Title: Business Development Manager
    Company: Global Solutions Co.
    Location: Mumbai, India
    About Us: Global Solutions Co. is a leading provider of enterprise software, helping businesses optimize their operations and achieve growth.
    Job Description: We are looking for an ambitious and results-driven Business Development Manager to identify new business opportunities, build strong client relationships, and drive revenue growth. You will be instrumental in expanding our market presence and achieving sales targets.
    Responsibilities:
    * Identify and research new business opportunities and target markets.
    * Develop and implement strategic sales plans.
    * Build and maintain strong, long-lasting client relationships.
    * Negotiate contracts and close agreements to maximize profits.
    * Prepare and deliver compelling presentations and proposals.
    * Collaborate with marketing and product teams to align strategies.
    * Monitor market trends and competitor activities.
    Requirements:
    * Bachelor's degree in Business, Marketing, or a related field.
    * 3+ years of proven experience in business development or sales.
    * Demonstrated ability to meet and exceed sales quotas.
    * Strong negotiation, presentation, and communication skills.
    * Proficiency in CRM software (e.g., Salesforce).
    * Strategic thinking and problem-solving abilities.
    """,
    "FINANCE": """
    Job Title: Financial Analyst
    Company: InvestRight Financial Services
    Location: Delhi, India
    About Us: InvestRight is a dynamic financial services firm providing investment banking, asset management, and advisory solutions to a diverse client base.
    Job Description: We are seeking a meticulous and analytical Financial Analyst to support our investment banking division. You will be responsible for conducting financial modeling, valuation, and analysis to guide strategic decision-making.
    Responsibilities:
    * Develop and maintain complex financial models for valuation and forecasting.
    * Conduct in-depth industry and company research.
    * Prepare detailed financial reports, presentations, and pitch books.
    * Assist in due diligence for mergers and acquisitions.
    * Analyze financial statements, market trends, and economic indicators.
    * Support senior analysts and managers in various financial projects.
    Requirements:
    * Bachelor's degree in Finance, Economics, Accounting, or a related quantitative field.
    * 1-3 years of experience in financial analysis, investment banking, or corporate finance.
    * Strong proficiency in Excel and financial modeling.
    * Knowledge of financial statements and accounting principles.
    * Excellent analytical, problem-solving, and communication skills.
    * CFA candidacy or certification is a plus.
    """,
    "ADVOCATE": """
    Job Title: Junior Legal Advocate
    Company: LexJuris Law Chambers
    Location: Chennai, India
    About Us: LexJuris Law Chambers is a reputable law firm specializing in corporate law, civil litigation, and intellectual property. We are committed to providing exceptional legal services.
    Job Description: We are seeking a motivated Junior Legal Advocate to join our litigation team. The ideal candidate will assist senior advocates in legal research, drafting, and court proceedings across various civil matters.
    Responsibilities:
    * Conduct comprehensive legal research and analysis.
    * Draft legal documents, including plaints, written statements, petitions, and appeals.
    * Assist in preparing for court hearings, trials, and arbitrations.
    * Attend court proceedings with senior advocates.
    * Manage case files and maintain accurate records.
    * Communicate with clients, opposing counsel, and court personnel.
    Requirements:
    * Bachelor of Laws (LLB) degree from a recognized institution.
    * Enrollment with the Bar Council of India.
    * 0-2 years of experience in litigation or legal practice.
    * Strong research, analytical, and drafting skills.
    * Excellent written and verbal communication.
    * Ability to work independently and as part of a team.
    """,
    "ACCOUNTANT": """
    Job Title: Staff Accountant
    Company: AccuBooks Accounting Services
    Location: Pune, India
    About Us: AccuBooks provides comprehensive accounting and taxation services to small and medium-sized businesses, ensuring financial compliance and strategic growth.
    Job Description: We are looking for a detail-oriented Staff Accountant to manage daily accounting operations and ensure the accuracy of financial records. You will be responsible for preparing financial statements, reconciling accounts, and assisting with audits.
    Responsibilities:
    * Prepare and examine accounting records, financial statements, and other financial reports.
    * Reconcile bank statements, general ledger accounts, and balance sheet accounts.
    * Process accounts payable and accounts receivable.
    * Assist with month-end and year-end closing procedures.
    * Ensure compliance with accounting standards and regulatory requirements.
    * Support internal and external audits.
    * Maintain accurate and organized financial documentation.
    Requirements:
    * Bachelor's degree in Accounting, Finance, or a related field.
    * 1-3 years of experience in an accounting role.
    * Proficiency in accounting software (e.g., Tally, QuickBooks, SAP) and MS Excel.
    * Strong understanding of GAAP (Generally Accepted Accounting Principles).
    * Excellent attention to detail and accuracy.
    * Good communication and organizational skills.
    """,
    "ENGINEERING": """
    Job Title: Mechanical Design Engineer
    Company: Innovate Engineering Solutions
    Location: Hyderabad, India
    About Us: Innovate Engineering Solutions is a leader in designing and developing advanced machinery and industrial systems for various sectors.
    Job Description: We are seeking a talented Mechanical Design Engineer to join our product development team. You will be responsible for conceptualizing, designing, and optimizing mechanical components and systems for new and existing products.
    Responsibilities:
    * Design mechanical systems and components using CAD software (e.g., SolidWorks, AutoCAD, CATIA).
    * Perform engineering calculations and simulations (FEA).
    * Create detailed technical drawings and specifications.
    * Conduct design reviews and collaborate with manufacturing teams.
    * Prototype, test, and validate designs.
    * Troubleshoot design-related issues and implement corrective actions.
    * Ensure designs meet performance, cost, and reliability targets.
    Requirements:
    * Bachelor's or Master's degree in Mechanical Engineering.
    * 3+ years of experience in mechanical design.
    * Proficiency in 3D CAD software and FEA tools.
    * Strong understanding of mechanical principles, materials, and manufacturing processes.
    * Experience with DFM (Design for Manufacturability) and DFA (Design for Assembly).
    * Excellent problem-solving and analytical skills.
    """,
    "CHEF": """
    Job Title: Sous Chef
    Company: The Grand Feast Restaurant
    Location: Goa, India
    About Us: The Grand Feast is a renowned fine-dining restaurant celebrated for its innovative culinary creations and exceptional guest experience.
    Job Description: We are seeking a skilled and passionate Sous Chef to support the Head Chef in all kitchen operations. You will be responsible for overseeing food preparation, maintaining quality standards, and managing kitchen staff.
    Responsibilities:
    * Assist the Head Chef in menu planning, recipe development, and food costing.
    * Oversee daily kitchen operations, including food preparation, cooking, and plating.
    * Ensure high standards of food quality, presentation, and consistency.
    * Manage kitchen inventory, ordering, and waste control.
    * Train, supervise, and motivate kitchen staff.
    * Maintain a clean, organized, and hygienic kitchen environment.
    * Adhere to all food safety and sanitation regulations.
    Requirements:
    * Culinary degree or equivalent professional training.
    * 3+ years of experience in a professional kitchen, with at least 1 year as a Sous Chef or equivalent.
    * Proven culinary skills and creativity.
    * Strong knowledge of various cooking techniques and cuisines.
    * Excellent leadership, communication, and organizational skills.
    * Ability to work under pressure in a fast-paced environment.
    """,
    "AVIATION": """
    Job Title: Aircraft Maintenance Technician
    Company: SkyWings Airlines
    Location: Delhi, India
    About Us: SkyWings Airlines is a leading regional airline committed to providing safe and reliable air travel with a focus on operational excellence.
    Job Description: We are looking for a certified Aircraft Maintenance Technician to perform scheduled and unscheduled maintenance on our fleet. You will be responsible for troubleshooting, repairing, and ensuring the airworthiness of aircraft systems and components.
    Responsibilities:
    * Perform routine inspections, maintenance, and repairs on aircraft components and systems.
    * Diagnose mechanical and electrical issues using specialized tools and equipment.
    * Conduct pre-flight and post-flight checks.
    * Adhere to all aviation regulations, safety procedures, and maintenance manuals.
    * Maintain accurate records of all maintenance activities.
    * Collaborate with engineers and flight crew to resolve technical issues.
    * Ensure timely completion of maintenance tasks to support flight schedules.
    Requirements:
    * Aircraft Maintenance Engineering (AME) license or equivalent certification.
    * 2+ years of experience in aircraft maintenance.
    * Strong knowledge of aircraft systems (mechanical, electrical, hydraulic).
    * Proficiency in using maintenance tools and diagnostic equipment.
    * Excellent problem-solving skills and attention to detail.
    * Ability to work flexible hours, including nights, weekends, and holidays.
    """,
    "FITNESS": """
    Job Title: Certified Fitness Trainer
    Company: EliteFit Gym & Studio
    Location: Bangalore, India
    About Us: EliteFit is a premium fitness facility offering personalized training, group classes, and wellness programs to help clients achieve their health goals.
    Job Description: We are seeking a highly motivated and certified Fitness Trainer to provide personalized coaching and lead engaging group fitness classes. You will inspire and guide clients towards a healthier lifestyle.
    Responsibilities:
    * Conduct fitness assessments and develop individualized exercise programs.
    * Provide one-on-one personal training sessions.
    * Lead dynamic and engaging group fitness classes (e.g., HIIT, Yoga, Zumba).
    * Educate clients on proper exercise techniques and nutrition.
    * Monitor client progress and adjust programs as needed.
    * Maintain a safe and motivating training environment.
    * Build strong relationships with clients and encourage consistent participation.
    Requirements:
    * Nationally recognized personal training certification (e.g., ACE, NASM, ACSM).
    * 1+ year of experience as a fitness trainer.
    * Strong knowledge of anatomy, exercise physiology, and nutrition.
    * Excellent communication, motivational, and interpersonal skills.
    * CPR/AED certification.
    * Passion for health and fitness.
    """,
    "SALES": """
    Job Title: Sales Executive
    Company: MarketMovers Pvt Ltd.
    Location: Mumbai, India
    About Us: MarketMovers is a leading distributor of FMCG products, with a vast network across urban and rural markets. We value customer relationships and aggressive market penetration.
    Job Description: We are looking for an energetic and target-driven Sales Executive to expand our client base and achieve sales objectives. You will be responsible for identifying leads, presenting products, and closing sales to contribute to our company's growth.
    Responsibilities:
    * Identify and qualify new sales leads through cold calling, networking, and referrals.
    * Present and demonstrate product features and benefits to prospective clients.
    * Negotiate sales terms and close deals.
    * Build and maintain strong, long-lasting customer relationships.
    * Achieve monthly and quarterly sales targets.
    * Prepare sales reports and forecasts.
    * Stay updated on product knowledge and market conditions.
    Requirements:
    * Bachelor's degree in Business, Marketing, or a related field (or equivalent experience).
    * 1-3 years of proven sales experience, preferably in FMCG or B2B sales.
    * Strong negotiation and persuasion skills.
    * Excellent verbal and written communication.
    * Results-oriented with a strong desire to meet and exceed targets.
    * Ability to travel as needed.
    """,
    "BANKING": """
    Job Title: Relationship Manager (Retail Banking)
    Company: Zenith Bank
    Location: Gurgaon, India
    About Us: Zenith Bank is a premier financial institution offering a full range of banking products and services to individual and corporate clients.
    Job Description: We are seeking a customer-focused Relationship Manager to nurture and grow our retail banking client portfolio. You will be responsible for understanding client needs, offering suitable banking products, and ensuring high levels of customer satisfaction.
    Responsibilities:
    * Manage a portfolio of retail banking clients, building strong relationships.
    * Identify client financial needs and cross-sell appropriate banking products (loans, accounts, investments).
    * Achieve individual and team sales targets.
    * Provide excellent customer service and resolve client inquiries efficiently.
    * Ensure compliance with banking regulations and internal policies.
    * Monitor market conditions and competitor offerings.
    * Maintain accurate client records and transaction details.
    Requirements:
    * Bachelor's degree in Finance, Business Administration, or a related field.
    * 2+ years of experience in retail banking, sales, or customer relationship management.
    * Strong understanding of banking products and services.
    * Excellent interpersonal, communication, and negotiation skills.
    * Customer-centric approach with a proven track record of sales.
    * Knowledge of KYC and AML regulations.
    """,
    "HEALTHCARE": """
    Job Title: Registered Nurse (RN)
    Company: Hopewell Hospital
    Location: Mumbai, India
    About Us: Hopewell Hospital is a leading multi-specialty healthcare facility dedicated to providing compassionate and high-quality patient care.
    Job Description: We are seeking a compassionate and skilled Registered Nurse to provide direct patient care across various departments. You will play a vital role in patient assessment, medication administration, and care coordination.
    Responsibilities:
    * Assess, plan, implement, and evaluate patient care plans.
    * Administer medications and treatments as prescribed by physicians.
    * Monitor patient vital signs and respond to changes in condition.
    * Maintain accurate and detailed patient records.
    * Educate patients and their families on health conditions and care instructions.
    * Collaborate with doctors, other nurses, and healthcare professionals.
    * Adhere to all hospital policies and nursing standards.
    Requirements:
    * Bachelor of Science in Nursing (BSN) or Diploma in Nursing.
    * Valid Registered Nurse (RN) license.
    * 1-3 years of experience in a hospital setting (e.g., Med-Surg, ICU, ER).
    * Strong clinical skills and critical thinking ability.
    * Excellent communication and interpersonal skills.
    * Ability to work in a fast-paced and demanding environment.
    * BLS/ACLS certification preferred.
    """,
    "CONSULTANT": """
    Job Title: Management Consultant (Junior)
    Company: Stratagem Consulting Group
    Location: Bangalore, India
    About Us: Stratagem Consulting Group helps organizations achieve strategic objectives through data-driven insights and innovative solutions across various industries.
    Job Description: We are looking for a bright and analytical Junior Management Consultant to support our client engagements. You will assist in research, data analysis, and developing strategic recommendations for businesses facing complex challenges.
    Responsibilities:
    * Conduct market research and industry analysis.
    * Collect, analyze, and interpret quantitative and qualitative data.
    * Develop financial models and business cases.
    * Prepare compelling presentations and reports for clients.
    * Assist senior consultants in developing strategic recommendations.
    * Participate in client meetings and workshops.
    * Contribute to problem-solving and solution design.
    Requirements:
    * Bachelor's or Master's degree in Business, Economics, Engineering, or a related analytical field.
    * 0-2 years of experience in consulting, business analysis, or a related role.
    * Strong analytical, problem-solving, and critical thinking skills.
    * Proficiency in MS Office Suite, especially Excel and PowerPoint.
    * Excellent written and verbal communication.
    * Ability to work effectively in a team-oriented, client-facing environment.
    """,
    "CONSTRUCTION": """
    Job Title: Civil Site Engineer
    Company: BuildStrong Constructions
    Location: Delhi NCR, India
    About Us: BuildStrong Constructions is a leading construction firm specializing in residential, commercial, and infrastructure projects, committed to quality and timely delivery.
    Job Description: We are seeking an experienced Civil Site Engineer to manage on-site construction activities. You will be responsible for supervising work, ensuring adherence to plans, and maintaining project timelines and quality standards.
    Responsibilities:
    * Oversee daily construction operations on site.
    * Interpret blueprints, drawings, and specifications.
    * Supervise construction workers and subcontractors.
    * Ensure compliance with safety regulations and quality control standards.
    * Monitor project progress, budget, and resource allocation.
    * Identify and resolve technical issues on site.
    * Prepare daily, weekly, and monthly progress reports.
    * Liaise with clients, architects, and other stakeholders.
    Requirements:
    * Bachelor's degree in Civil Engineering.
    * 3+ years of experience as a Site Engineer in building construction.
    * Strong knowledge of construction methods, materials, and safety procedures.
    * Proficiency in AutoCAD and project management software.
    * Excellent leadership, problem-solving, and communication skills.
    * Ability to work effectively under pressure.
    """,
    "PUBLIC-RELATIONS": """
    Job Title: Public Relations Executive
    Company: MediaLink Communications
    Location: Mumbai, India
    About Us: MediaLink Communications is a full-service PR agency dedicated to building and managing the reputations of our diverse clients across various industries.
    Job Description: We are looking for a creative and proactive Public Relations Executive to support our client accounts. You will be responsible for media relations, content creation, and executing PR campaigns to enhance brand visibility and reputation.
    Responsibilities:
    * Develop and implement PR strategies and campaigns.
    * Draft press releases, media kits, speeches, and other PR materials.
    * Cultivate and maintain strong relationships with media contacts, journalists, and influencers.
    * Monitor media coverage and industry trends.
    * Organize press conferences, media events, and product launches.
    * Manage social media content and engagement strategies.
    * Prepare client reports on PR activities and outcomes.
    Requirements:
    * Bachelor's degree in Public Relations, Communications, Journalism, or Marketing.
    * 1-3 years of experience in public relations, corporate communications, or media.
    * Excellent written and verbal communication and storytelling skills.
    * Strong media relations skills and an understanding of the media landscape.
    * Ability to work under pressure and manage multiple projects.
    * Creativity and a keen eye for detail.
    """,
    "HR": """
    Job Title: HR Executive
    Company: PeopleConnect Solutions
    Location: Pune, India
    About Us: PeopleConnect Solutions is a growing HR consulting firm providing recruitment, talent management, and HR advisory services to various industries.
    Job Description: We are seeking a dedicated HR Executive to support various human resources functions. You will be involved in recruitment, employee relations, HR administration, and ensuring a positive employee experience.
    Responsibilities:
    * Manage the end-to-end recruitment process (sourcing, screening, interviewing, onboarding).
    * Assist with employee onboarding and offboarding procedures.
    * Maintain accurate employee records and HR databases.
    * Address employee inquiries and provide support on HR policies.
    * Assist in performance management processes and training initiatives.
    * Ensure compliance with labor laws and company policies.
    * Support HR projects and initiatives as needed.
    Requirements:
    * Bachelor's degree in Human Resources, Business Administration, or a related field.
    * 1-3 years of experience in an HR role.
    * Strong knowledge of HR best practices and labor laws.
    * Excellent interpersonal and communication skills.
    * Proficiency in HRIS (Human Resources Information Systems) and MS Office.
    * Empathy and strong problem-solving abilities.
    """,
    "DESIGNER": """
    Job Title: UI/UX Designer
    Company: PixelCraft Studios
    Location: Bangalore, India
    About Us: PixelCraft Studios is a dynamic digital agency specializing in creating intuitive and visually stunning user experiences for web and mobile applications.
    Job Description: We are looking for a creative and user-centered UI/UX Designer to craft exceptional digital experiences. You will be responsible for user research, wireframing, prototyping, and designing intuitive interfaces.
    Responsibilities:
    * Conduct user research, usability testing, and analyze user feedback.
    * Create wireframes, storyboards, user flows, and site maps.
    * Develop high-fidelity prototypes and mockups.
    * Design intuitive and aesthetically pleasing user interfaces for web and mobile.
    * Collaborate with product managers and developers to ensure design implementation.
    * Ensure design consistency across all platforms and adherence to brand guidelines.
    * Stay updated with the latest UI/UX trends and tools.
    Requirements:
    * Bachelor's degree in Design, Human-Computer Interaction, or a related field.
    * 2+ years of experience in UI/UX design.
    * Proficiency in design tools such as Figma, Sketch, Adobe XD, or Adobe Creative Suite.
    * Strong portfolio showcasing UI/UX design projects.
    * Understanding of user-centered design principles and usability best practices.
    * Excellent communication and problem-solving skills.
    """,
    "ARTS": """
    Job Title: Graphic Designer
    Company: Creative Canvas Agency
    Location: Delhi, India
    About Us: Creative Canvas Agency is a boutique design firm passionate about delivering innovative and impactful visual solutions for brands across various industries.
    Job Description: We are seeking a talented and imaginative Graphic Designer to create compelling visual content for marketing, branding, and digital platforms. You will be responsible for translating concepts into visually engaging designs.
    Responsibilities:
    * Design and produce creative assets for digital and print (logos, brochures, social media graphics, websites).
    * Collaborate with marketing and content teams to understand design requirements.
    * Develop concepts, graphics, and layouts for product illustrations, company logos, and websites.
    * Ensure brand consistency across all visual communications.
    * Manage multiple design projects from concept to completion.
    * Stay up-to-date with industry trends and design software.
    * Prepare design files for production and print.
    Requirements:
    * Bachelor's degree or diploma in Graphic Design, Fine Arts, or a related field.
    * 2+ years of professional graphic design experience.
    * Proficiency in Adobe Creative Suite (Photoshop, Illustrator, InDesign).
    * Strong portfolio showcasing diverse design projects.
    * Excellent understanding of design principles, typography, and color theory.
    * Creativity, attention to detail, and ability to meet deadlines.
    """,
    "TEACHER": """
    Job Title: Primary School Teacher
    Company: Bright Future School
    Location: Kolkata, India
    About Us: Bright Future School is a reputed educational institution committed to providing a nurturing and stimulating learning environment for young minds.
    Job Description: We are seeking a passionate and dedicated Primary School Teacher to inspire and educate students in grades 1-5. You will be responsible for creating engaging lesson plans, fostering a positive classroom environment, and assessing student progress.
    Responsibilities:
    * Plan, prepare, and deliver engaging lessons according to the curriculum.
    * Create a positive, inclusive, and stimulating learning environment.
    * Assess student progress, provide constructive feedback, and maintain records.
    * Communicate effectively with parents regarding student development and concerns.
    * Implement various teaching methods to accommodate diverse learning styles.
    * Organize and supervise classroom activities and field trips.
    * Participate in school events and professional development programs.
    Requirements:
    * Bachelor's degree in Education (B.Ed) or equivalent teaching qualification.
    * 1-3 years of experience teaching in a primary school setting.
    * Strong knowledge of child development and pedagogical approaches.
    * Excellent classroom management and communication skills.
    * Patience, creativity, and a genuine love for teaching young children.
    * Ability to integrate technology into the classroom.
    """,
    "APPAREL": """
    Job Title: Fashion Designer (Apparel)
    Company: StyleSense Fashion House
    Location: Delhi, India
    About Us: StyleSense Fashion House is an emerging brand known for its contemporary and sustainable apparel collections for the modern consumer.
    Job Description: We are looking for a creative and innovative Fashion Designer to join our apparel design team. You will be responsible for conceptualizing, sketching, and developing new clothing lines from initial idea to final production.
    Responsibilities:
    * Research fashion trends, market demands, and consumer preferences.
    * Develop mood boards, sketches, and technical drawings for new collections.
    * Select fabrics, trims, and embellishments.
    * Collaborate with pattern makers and sample room for sample development.
    * Oversee fittings and make necessary adjustments to designs.
    * Ensure designs are aligned with brand aesthetics and production capabilities.
    * Work closely with merchandising and production teams.
    Requirements:
    * Bachelor's degree in Fashion Design or a related field.
    * 2+ years of experience in apparel design, with a strong portfolio.
    * Proficiency in design software (e.g., Adobe Illustrator, Photoshop, CAD tools).
    * Strong understanding of garment construction, textiles, and fashion trends.
    * Excellent sketching and illustration skills.
    * Creativity, attention to detail, and ability to work in a fast-paced environment.
    """,
    "DIGITAL-MEDIA": """
    Job Title: Digital Content Creator
    Company: Visionary Digital Hub
    Location: Bangalore, India
    About Us: Visionary Digital Hub is a dynamic agency specializing in creating engaging digital content and marketing strategies for diverse clients.
    Job Description: We are seeking a versatile Digital Content Creator to produce compelling visual and written content across various digital platforms. You will be responsible for ideating, scripting, shooting, editing, and publishing engaging media.
    Responsibilities:
    * Develop creative concepts and storyboards for digital content (videos, graphics, articles).
    * Write engaging copy for social media posts, blog articles, and website content.
    * Shoot and edit high-quality videos and photos.
    * Design visually appealing graphics and animations.
    * Manage content calendars and publishing schedules.
    * Analyze content performance and optimize strategies.
    * Stay updated on digital media trends and platform algorithms.
    Requirements:
    * Bachelor's degree in Digital Media, Mass Communication, Journalism, or a related field.
    * 1-3 years of experience in digital content creation.
    * Proficiency in video editing software (e.g., Adobe Premiere Pro, Final Cut Pro) and graphic design tools (e.g., Adobe Photoshop, Illustrator).
    * Strong writing and storytelling skills.
    * Experience with social media platforms and content management systems.
    * Creativity, attention to detail, and ability to work independently.
    """,
    "AGRICULTURE": """
    Job Title: Agricultural Field Officer
    Company: GreenHarvest AgriTech
    Location: Rural Regions (India, specific states TBD)
    About Us: GreenHarvest AgriTech is committed to empowering farmers with sustainable agricultural practices, modern technology, and market access to enhance productivity and livelihoods.
    Job Description: We are seeking a proactive Agricultural Field Officer to work directly with farmers, providing guidance on modern farming techniques, crop management, and sustainable practices. You will be a key link between our organization and the farming community.
    Responsibilities:
    * Visit farms and interact directly with farmers to understand their needs.
    * Provide technical advice on crop rotation, pest control, irrigation, and fertilization.
    * Promote and educate farmers on sustainable agricultural practices and new technologies.
    * Collect data on crop yield, soil health, and farming challenges.
    * Organize farmer training programs and workshops.
    * Assist in the distribution of seeds, fertilizers, and other agricultural inputs.
    * Prepare field reports and contribute to agricultural development plans.
    Requirements:
    * Bachelor's degree in Agriculture, Agronomy, or a related field.
    * 1-3 years of experience in agricultural extension, field work, or farming.
    * Strong knowledge of crop science, soil management, and modern farming techniques.
    * Excellent communication and interpersonal skills, especially with rural communities.
    * Ability to travel extensively to rural areas.
    * Proficiency in local languages is highly preferred.
    """,
    "AUTOMOBILE": """
    Job Title: Automotive Service Technician
    Company: AutoCare Solutions
    Location: Chennai, India
    About Us: AutoCare Solutions is a well-established automotive service center offering comprehensive repair and maintenance services for various vehicle brands.
    Job Description: We are seeking a skilled Automotive Service Technician to perform diagnostics, maintenance, and repairs on a wide range of vehicles. You will be responsible for ensuring vehicle safety, performance, and customer satisfaction.
    Responsibilities:
    * Diagnose mechanical and electrical issues in vehicles using diagnostic equipment.
    * Perform routine maintenance (oil changes, tire rotations, brake inspections).
    * Conduct complex repairs on engines, transmissions, braking systems, and suspensions.
    * Test drive vehicles to ensure proper functionality after repairs.
    * Communicate clearly with service advisors and customers regarding vehicle issues and repair recommendations.
    * Maintain accurate records of services performed.
    * Adhere to all safety regulations and workshop standards.
    Requirements:
    * ITI/Diploma in Automotive Technology or equivalent vocational training.
    * 2+ years of experience as an automotive technician or mechanic.
    * Strong knowledge of vehicle systems and diagnostic procedures.
    * Proficiency in using automotive tools and equipment.
    * Excellent problem-solving skills and attention to detail.
    * Valid driver's license.
    """,
    "BPO": """
    Job Title: Customer Service Representative (BPO)
    Company: ConnectGlobal Services
    Location: Noida, India
    About Us: ConnectGlobal Services is a leading Business Process Outsourcing (BPO) firm providing exceptional customer support and back-office solutions to global clients.
    Job Description: We are looking for an enthusiastic Customer Service Representative to join our BPO team. You will be the first point of contact for customers, providing support, resolving inquiries, and ensuring a positive customer experience across various channels.
    Responsibilities:
    * Handle inbound and outbound customer calls, emails, and chat inquiries.
    * Provide accurate information and resolve customer issues efficiently and courteously.
    * Document all customer interactions and resolutions in the CRM system.
    * Escalate complex issues to the appropriate department when necessary.
    * Adhere to communication scripts and quality standards.
    * Maintain a high level of customer satisfaction.
    * Collaborate with team members to achieve daily targets.
    Requirements:
    * High school diploma or equivalent; Bachelor's degree preferred.
    * 0-2 years of experience in customer service, call center, or BPO environment.
    * Excellent verbal and written communication skills in English (and regional languages if required).
    * Strong active listening and problem-solving abilities.
    * Proficiency in basic computer skills and navigating software applications.
    * Ability to work in a fast-paced, target-driven environment.
    * Flexibility to work in shifts (day/night).
    """
}

# Convert dictionary to DataFrame
job_data = []
for category, desc in job_descriptions.items():
    # Extract job title from the text (assuming it's the first line after "Job Title:")
    job_title = desc.split("Job Title:")[1].split("\n")[0].strip()
    job_data.append({'category': category, 'job_title': job_title, 'description': desc})


job_descriptions_df = pd.DataFrame(job_data)
# Apply the same preprocessing pipeline to job descriptions
job_descriptions_df['cleaned_description'] = job_descriptions_df['description'].apply(clean_text)
job_descriptions_df['processed_description'] = job_descriptions_df['cleaned_description'].apply(preprocess_text)
job_descriptions_df['lemmatized_description'] = job_descriptions_df['processed_description'].apply(lemmatize_text)




import streamlit as st


# Streamlit-cached getter for vectorizer and resume vectors
@st.cache_resource(show_spinner=False)
def get_vectorizer_and_matrix():
    from sklearn.feature_extraction.text import TfidfVectorizer
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_vectorizer.fit(df['lemmatized_resume'])
    resume_vectors = tfidf_vectorizer.transform(df['lemmatized_resume'])
    return tfidf_vectorizer, resume_vectors

# Expose as module-level variables for safe import
tfidf_vectorizer, resume_vectors = get_vectorizer_and_matrix()




def find_resumes_for_job(job_description, dataframe, vectorizer, resume_matrix, num_matches=5):
    """
    Takes a job description and finds the top matching resumes from the dataset.
    Returns a list of dicts with similarity score, category, and resume snippet.
    """
    lemmatized_job = lemmatize_text(preprocess_text(clean_text(job_description)))
    job_vector = vectorizer.transform([lemmatized_job])
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np
    cosine_sim_scores = cosine_similarity(job_vector, resume_matrix).flatten()
    top_resume_indices = np.argsort(cosine_sim_scores)[::-1][:num_matches]
    results = []
    for rank, resume_idx in enumerate(top_resume_indices):
        resume_category = dataframe['Category'].iloc[resume_idx]
        similarity_score = cosine_sim_scores[resume_idx]
        original_resume_text = dataframe['lemmatized_resume'].iloc[resume_idx]
        results.append({
            "Rank": rank + 1,
            "Similarity Score": round(similarity_score, 4),
            "Category": resume_category,
            "Resume Snippet": original_resume_text[:200] + "..."
        })
    return results

# --- Improved: Match resumes by job category ---
def find_resumes_for_job_category(category, job_descriptions_df, resumes_df, vectorizer, resume_matrix, num_matches=5):
    row = job_descriptions_df[job_descriptions_df['category'] == category]
    if row.empty:
        return []
    job_description = row['description'].values[0]
    return find_resumes_for_job(job_description, resumes_df, vectorizer, resume_matrix, num_matches)