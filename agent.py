import os
import json
import datetime
import time
import html
import re
import subprocess
import faiss
import pickle
from typing import List, Dict
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from retrievers import FAISSRetriever
from flask import Flask
import socket
import threading
from pyngrok import ngrok
from flask import Flask, request, jsonify, render_template, send_file
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()


# ---------------- FAISS Import Check ----------------
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False


class IPCCLLMAgent:
    def __init__(self):
        self.session_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.history_dir = os.path.join(os.getcwd(), "history")
        self.conversation_history = self.load_conversation_history()
        self.ipcc_reports = {
            'all': {'name': 'All IPCC Reports', 'color': ''},
            'srocc': {'name': 'SROCC Summary for Policymakers (2019)', 'color': ''},
            'ar6_syr_full': {'name': 'AR6 Synthesis Report Full Volume (2023)', 'color': ''},
            'ar6_syr_slides': {'name': 'AR6 Synthesis Report Slide Deck (2023)', 'color': ''},
            'ar6_wgii_ts': {'name': 'AR6 WGII Technical Summary (2022)', 'color': ''},
            'ar6_wgiii': {'name': 'AR6 WGIII Full Report (2022)', 'color': ''},
            'sr15': {'name': 'SR15 1.5°C Full Report (2018)', 'color': ''},
            'srccl': {'name': 'SRCCL Full Report (2019)', 'color': ''},
        }
        self.llm_models = {
            'deepseek': {'name': 'DeepSeek-R1-Distill-Llama-70B', 'provider': 'Groq'},
            'llama': {'name': 'Llama-3.3-70B-Versatile', 'provider': 'Groq'},
            'gemma2': {'name': 'Gemma2-9B-IT', 'provider': 'Groq'},
            'qwen': {'name': 'Qwen3-32B', 'provider': 'Groq'},
            'compound-beta-mini': {'name': 'Compound-Beta-Mini', 'provider': 'Groq'},
            'mock': {'name': 'Mock AI (Demo)', 'provider': 'Local'}
        }
        self.setup_api_clients()
        self.ipcc_knowledge = self.load_ipcc_knowledge()
        self.faiss_retrievers = self.setup_faiss_retrievers()

    def setup_api_clients(self):
      self.groq_client_llama = None   # disabled for now (waitlist)
      self.groq_client_deepseek = None
      self.groq_client_gemma2 = None
      self.groq_client_qwen = None
      self.groq_client_compound_beta_mini = None

      # ✅ Use environment variable only (no userdata)
      GROQ_API_KEY = os.getenv("GROQ_API_KEY")   # from your .env

      if not GROQ_API_KEY:
        print("⚠️ No Groq API key found. Please set GROQ_API_KEY in your .env file")
      else:
        print("✅ Groq API key loaded")


    def setup_faiss_retrievers(self):
      if not FAISS_AVAILABLE:
          print("FAISS not available. Skipping FAISS setup.")
          return {}

      # Use a local folder in your project root
      base_path = os.path.join(os.getcwd(), "faiss_data")

      retrievers = {}
      file_mapping = {
          'srocc': {
              'index': '01_SROCC_SPM_FINAL_index.bin',
              'texts': '01_SROCC_SPM_FINAL_texts.pkl'
          },
          'ar6_syr_full': {
              'index': 'IPCC_AR6_SYR_FullVolume_index.bin',
              'texts': 'IPCC_AR6_SYR_FullVolume_texts.pkl'
          },
          'ar6_syr_slides': {
              'index': 'IPCC_AR6_SYR_SlideDeck_index.bin',
              'texts': 'IPCC_AR6_SYR_SlideDeck_texts.pkl'
          },
          'ar6_wgii_ts': {
              'index': 'IPCC_AR6_WGII_TechnicalSummary_index.bin',
              'texts': 'IPCC_AR6_WGII_TechnicalSummary_texts.pkl'
          },
          'ar6_wgiii': {
              'index': 'IPCC_AR6_WGIII_FullReport_index.bin',
              'texts': 'IPCC_AR6_WGIII_FullReport_texts.pkl'
          },
          'sr15': {
              'index': 'SR15_Full_Report_LR_index.bin',
              'texts': 'SR15_Full_Report_LR_texts.pkl'
          },
          'srccl': {
              'index': 'SRCCL_Full_Report_index.bin',
              'texts': 'SRCCL_Full_Report_texts.pkl'
          }
      }

      try:
          embed_model = HuggingFaceEmbeddings(model_name="mixedbread-ai/mxbai-embed-large-v1")
          print("Embedding model initialized")

          for report_key in self.ipcc_reports.keys():
              if report_key == 'all':
                  continue
              if report_key not in file_mapping:
                  print(f"Warning: No FAISS file mapping for {report_key}")
                  continue

              index_filename = file_mapping[report_key]['index']
              texts_filename = file_mapping[report_key]['texts']

              index_path = os.path.join(base_path, index_filename)
              texts_path = os.path.join(base_path, texts_filename)

              print(f"Checking FAISS files for {report_key}:")
              print(f"  Index file: {index_path} exists: {os.path.exists(index_path)}")
              print(f"  Texts file: {texts_path} exists: {os.path.exists(texts_path)}")

              if os.path.exists(index_path) and os.path.exists(texts_path):
                  try:
                      retriever = FAISSRetriever(report_key, index_path, texts_path, embed_model, k=5)
                      retrievers[report_key] = retriever
                      print(f"FAISS retriever for {report_key} loaded successfully")
                  except Exception as e:
                      print(f"Error loading FAISS retriever for {report_key}: {str(e)}")
              else:
                  print(f"⚠️ FAISS files for {report_key} not found in {base_path}")

          return retrievers

      except Exception as e:
          print(f"Error setting up FAISS retrievers: {str(e)}")
          return {}

    def load_ipcc_knowledge(self):
        return {
            'srocc_summary': {
                'content': """# SROCC Summary for Policymakers: Key Findings
- **Ocean Warming**: Oceans have absorbed 90% of excess heat since 1970.
- **Sea Level Rise**: Global mean sea level rising at 3.7 mm/year, accelerating.
- **Glacier and Ice Loss**: Arctic sea ice declining; Greenland and Antarctic ice sheets losing mass.
- **Marine Ecosystems**: Coral bleaching and fishery declines due to warming and acidification.
- **Coastal Risks**: Increased flooding and erosion affecting millions by 2100.
- **Adaptation Needs**: Enhanced coastal defenses and ecosystem restoration.""",
                'sources': ['SROCC SPM (2019)']
            },
            'sr15_summary': {
                'content': """# SR15 1.5°C Full Report: Key Findings
- **1.5°C vs. 2°C**: Half a degree reduces severe impacts significantly.
- **Carbon Budget**: ~420 GtCO₂ remaining for 1.5°C (50% chance, 2018).
- **Emission Cuts**: 45% reduction by 2030, net zero by 2050.
- **Impacts**: Lower risks to ecosystems, health, and food security at 1.5°C.
- **Solutions**: Rapid energy transition, reforestation, and carbon capture.""",
                'sources': ['SR15 Full Report (2018)']
            },
            'srccl_summary': {
                'content': """# SRCCL Full Report: Key Findings
- **Land Degradation**: 23% of global land degraded, reducing carbon sinks.
- **Food Security**: Climate change exacerbates hunger; 821 million undernourished.
- **Deforestation**: Contributes 11% of GHG emissions.
- **Solutions**: Sustainable land management, dietary shifts, and reforestation.
- **Co-benefits**: Improved biodiversity, soil health, and livelihoods.""",
                'sources': ['SRCCL Full Report (2019)']
            },
            'ar6_wgii_summary': {
                'content': """# AR6 WGII Technical Summary: Key Findings
- **Vulnerable Populations**: 3.3–3.6 billion people in high-risk areas.
- **Ecosystem Impacts**: 14% of species at high extinction risk at 1.5°C.
- **Health Risks**: Increased heat-related mortality and disease spread.
- **Adaptation Gaps**: Current measures insufficient for 2°C scenarios.
- **Solutions**: Climate-resilient development and ecosystem-based adaptation.""",
                'sources': ['AR6 WGII Technical Summary (2022)']
            },
            'ar6_wgiii_summary': {
                'content': """# AR6 WGIII Full Report: Key Findings
- **Emission Trends**: GHG emissions rose 54% from 1990 to 2019.
- **1.5°C Pathway**: Peak emissions by 2025, 43% cut by 2030.
- **Sector Solutions**: Renewables, electrification, and efficiency improvements.
- **Costs**: Net-zero by 2050 achievable with 2–3% GDP investment.
- **Policy Needs**: Carbon pricing, subsidies reform, and just transitions.""",
                'sources': ['AR6 WGIII Full Report (2022)']
            }
        }

    def load_conversation_history(self):
        os.makedirs(self.history_dir, exist_ok=True)
        history_file = os.path.join(self.history_dir, f"history_{self.session_id}.json")
        if os.path.exists(history_file):
            try:
                with open(history_file, 'r') as f:
                    history = json.load(f)
                print(f"Loaded conversation history for session {self.session_id}")
                return history
            except Exception as e:
                print(f"Error loading history for session {self.session_id}: {str(e)}")
                return []
        return []

    def save_conversation_history(self):
        history_file = os.path.join(self.history_dir, f"history_{self.session_id}.json")
        try:
            with open(history_file, 'w') as f:
                json.dump(self.conversation_history, f, indent=2)
            print(f"Saved conversation history for session {self.session_id}")
        except Exception as e:
            print(f"Error saving history for session {self.session_id}: {str(e)}")

    def get_session_list(self):
        os.makedirs(self.history_dir, exist_ok=True)
        sessions = []
        for filename in os.listdir(self.history_dir):
            if filename.startswith("history_") and filename.endswith(".json"):
                session_id = filename.replace("history_", "").replace(".json", "")
                sessions.append(session_id)
        return sorted(sessions, reverse=True)

    def switch_session(self, session_id: str):
        self.session_id = session_id
        self.conversation_history = self.load_conversation_history()
        return self.conversation_history

    def new_session(self):
        self.session_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.conversation_history = []
        self.save_conversation_history()
        print(f"Started new session: {self.session_id}")
        return self.conversation_history

    def format_response(self, content: str, sources: List[str] = None, report_focus: str = 'all'):
        # Escape content to prevent HTML injection
        content = html.escape(content)
        # Convert basic Markdown-like formatting to HTML
        lines = content.split('\n')
        html_lines = []
        in_list = False
        for line in lines:
            line = line.strip()
            if not line:
                if in_list:
                    html_lines.append('</ul>')
                    in_list = False
                html_lines.append('<p></p>')
                continue
            if line.startswith('**') and line.endswith('**'):
                text = line[2:-2].strip()
                html_lines.append(f'<strong>{text}</strong>')
            elif line.startswith('- '):
                if not in_list:
                    html_lines.append('<ul>')
                    in_list = True
                html_lines.append(f'<li>{line[2:].strip()}</li>')
            else:
                if in_list:
                    html_lines.append('</ul>')
                    in_list = False
                html_lines.append(f'<p>{line}</p>')
        if in_list:
            html_lines.append('</ul>')
        content_html = ''.join(html_lines)
        # Build the full HTML response
        sources_html = ''
        if sources:
            sources_items = ''.join(f'<li>{html.escape(source)}</li>' for source in sources)
            sources_html = f'<ul class="sources-list">{sources_items}</ul>'
        return f"""
        <div class="response-content">
            <strong>Report Focus: {html.escape(self.ipcc_reports[report_focus]['name'])}</strong>
            {content_html}
            {sources_html}
        </div>
        """

    def get_mock_response(self, message: str, report_focus: str):
        print(f"Debug: Using mock response for message: {message}, report_focus: {report_focus}")
        message_lower = message.lower()
        if report_focus == 'srocc' or 'srocc' in message_lower or 'ocean' in message_lower:
            knowledge = self.ipcc_knowledge['srocc_summary']
            return knowledge['content'], knowledge['sources']
        elif report_focus == 'sr15' or '1.5' in message_lower or 'sr15' in message_lower:
            knowledge = self.ipcc_knowledge['sr15_summary']
            return knowledge['content'], knowledge['sources']
        elif report_focus == 'srccl' or 'land' in message_lower or 'srccl' in message_lower:
            knowledge = self.ipcc_knowledge['srccl_summary']
            return knowledge['content'], knowledge['sources']
        elif report_focus == 'ar6_wgii_ts' or 'impacts' in message_lower or 'wgii' in message_lower:
            knowledge = self.ipcc_knowledge['ar6_wgii_summary']
            return knowledge['content'], knowledge['sources']
        elif report_focus == 'ar6_wgiii' or 'mitigation' in message_lower or 'wgiii' in message_lower:
            knowledge = self.ipcc_knowledge['ar6_wgiii_summary']
            return knowledge['content'], knowledge['sources']
        elif report_focus in ['ar6_syr_full', 'ar6_syr_slides'] or 'synthesis' in message_lower:
            return ("Placeholder: AR6 Synthesis Report (Full or Slides) summary not available in mock mode. "
                    "Please select an LLM to access FAISS data."), ['Mock Response']
        else:
            return """I can help with IPCC reports! Try asking about:
- SROCC ocean and cryosphere findings
- SR15 1.5°C pathways
- SRCCL land use impacts
- AR6 WGII impacts and adaptation
- AR6 WGIII mitigation strategies""", ['IPCC Knowledge Base']

    async def clean_response(self, content: str) -> str:
        print(f"Debug: Raw response before cleaning: {content}")
        cleaned = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL)
        cleaned = re.sub(r'<reasoning>.*?</reasoning>', '', cleaned, flags=re.DOTALL)
        cleaned = re.sub(r'<[^>]+>', '', cleaned)
        cleaned = cleaned.strip()
        print(f"Debug: Cleaned response: {cleaned}")
        return cleaned

    def requires_web_search(self, message: str) -> bool:
        message_lower = message.lower()
        web_search_keywords = [
            'search', 'web', 'internet', 'online', 'recent', 'latest', 'current',
            'news', 'update', 'real-time', 'live data', 'fetch'
        ]
        return any(keyword in message_lower for keyword in web_search_keywords)

    async def call_llm_api(self, messages: List[Dict], model: str, report_focus: str):
        print(f"Debug: Calling LLM API with model={model}, report_focus={report_focus}")
        if model == 'mock':
            time.sleep(1)
            user_message = messages[-1]['content']
            return self.get_mock_response(user_message, report_focus)

        report_examples = {
            'srocc': '[SROCC Summary for Policymakers (2019)] Oceans have absorbed 90% of excess heat since 1970.',
            'ar6_syr_full': '[AR6 Synthesis Report Full Volume (2023)] Global warming is projected to reach 1.5°C by 2030.',
            'ar6_syr_slides': '[AR6 Synthesis Report Slide Deck (2023)] Emissions must peak by 2025 for 1.5°C pathways.',
            'ar6_wgii_ts': '[AR6 WGII Technical Summary (2022)] 3.3–3.6 billion people are in high-risk areas.',
            'ar6_wgiii': '[AR6 WGIII Full Report (2022)] GHG emissions rose 54% from 1990 to 2019.',
            'sr15': '[SR15 1.5°C Full Report (2018)] 45% emission cuts needed by 2030 for 1.5°C.',
            'srccl': '[SRCCL Full Report (2019)] 23% of global land is degraded.'
        }

        system_prompt = f"""You are an expert IPCC climate reports analyst. Provide accurate, science-based responses using only the provided context from the specified IPCC report: {self.ipcc_reports[report_focus]['name']}. Use the conversation history to maintain context and avoid repeating information unnecessarily.

Key points:
- Use specific data, figures, and projections from the context.
- Cite the report title for each piece of information using the format: '[{self.ipcc_reports[report_focus]['name']}]'.
- Example: {report_examples.get(report_focus, '[Report Title] Example data.')}
- Do not include information from other reports or generalize beyond the provided context.
- Explain complex concepts clearly for policymakers and general audiences.
- Highlight policy implications and actionable insights.
- Note uncertainties and ranges in projections."""

        deepseek_qwen_prompt = f"""You are an expert on IPCC climate reports. Answer accurately using only the provided context from: {self.ipcc_reports[report_focus]['name']}. Use the conversation history to maintain context and avoid repetition. Use specific data, figures, and projections, citing the report title for each piece of information as '[{self.ipcc_reports[report_focus]['name']}]'. Example: {report_examples.get(report_focus, '[Report Title] Example data.')}. Do not generalize beyond the context. Explain clearly for policymakers and general audiences."""

        web_search_prompt = """You are an expert IPCC climate reports assistant with access to a web search tool powered by Tavily. When the user requests recent or real-time data (e.g., 'search the web,' 'latest news,' 'current trends'), use the web search tool to fetch up-to-date information. Cite web sources with URLs in the format '[Source: <URL>]'. For IPCC report queries, use the provided context and cite the report title. Use the conversation history to maintain context and avoid repetition. If no web search is needed, rely solely on the context.

Key points:
- Use web search only when explicitly requested or when the query involves recent events or data not in the context.
- Include at least one relevant IPCC report fact from the context, cited as '[{self.ipcc_reports[report_focus]['name']}]', unless the query is entirely non-IPCC.
- Explain results clearly for policymakers and general audiences.
- Note uncertainties in web data and prioritize reputable sources (e.g., .edu, .gov, .org)."""

        groq_clients = {
            'deepseek': self.groq_client_deepseek,
            'llama': self.groq_client_llama,
            'gemma2': self.groq_client_gemma2,
            'qwen': self.groq_client_qwen,
            'compound-beta-mini': self.groq_client_compound_beta_mini
        }

        user_message = messages[-1]['content']
        use_web_search = model == 'compound-beta-mini' and self.requires_web_search(user_message)
        print(f"Debug: User message: {user_message}, use_web_search: {use_web_search}")

        if not self.faiss_retrievers and not use_web_search:
            print("Debug: No FAISS retrievers available, falling back to mock mode")
            return self.get_mock_response(user_message, report_focus)

        try:
            context_parts = []
            retrieved_reports = set()
            docs = []
            web_sources = []
            if not use_web_search:
                print(f"Debug: Retrieving FAISS documents for report_focus={report_focus}")
                if report_focus == 'all':
                    for report_key, retriever in self.faiss_retrievers.items():
                        docs.extend(retriever._get_relevant_documents(user_message))
                else:
                    retriever = self.faiss_retrievers.get(report_focus)
                    if retriever:
                        docs = retriever._get_relevant_documents(user_message)
                    else:
                        print(f"Debug: No retriever for report_focus={report_focus}, falling back to mock")
                        return self.get_mock_response(user_message, report_focus)

                for doc in docs:
                    report_key = doc.metadata.get('report_key', 'Unknown')
                    retrieved_reports.add(report_key)
                    report_name = self.ipcc_reports.get(report_key, {'name': report_key})['name']
                    context_parts.append(f"[{report_name}] {doc.page_content}")
                context = "\n\n".join(context_parts)
                report_names = [self.ipcc_reports.get(key, {'name': key})['name'] for key in retrieved_reports]
                print(f"Debug: Retrieved {len(docs)} documents from reports: {retrieved_reports}")
            else:
                context = ""
                report_names = [self.ipcc_reports.get(report_focus, {'name': report_focus})['name']]
                print("Debug: Web search mode, no FAISS context retrieved")

            if report_focus != 'all' and retrieved_reports and any(key != report_focus for key in retrieved_reports):
                print(f"Warning: Retrieved unexpected report_keys: {retrieved_reports} for report_focus={report_focus}")

            report_title_context = ""
            if report_focus == 'all' and retrieved_reports and not use_web_search:
                report_title_context = (
                    "The following IPCC reports are included in the context below. "
                    "Always attribute data to the specific report title, using the format '[Report Title]'. "
                    "For example: '[AR6 WGII Technical Summary (2022)] 3.3–3.6 billion people are vulnerable.'\n" +
                    "\n".join(f"- {name}" for name in report_names) +
                    "\n\n"
                )

            prompt_messages = [
                {"role": "system", "content": (
                    f"{web_search_prompt if use_web_search else deepseek_qwen_prompt if model in ['deepseek', 'qwen', 'compound-beta-mini'] else system_prompt}\n\n"
                    f"{report_title_context}Context:\n{context}"
                )}
            ] + messages

            print(f"Debug: Prompt messages for {model}: {prompt_messages}")

            if model in groq_clients:
                client = groq_clients.get(model)
                if client:
                    try:
                        if use_web_search:
                            print("Debug: Invoking Groq with web search")
                            response = client.invoke(
                                prompt_messages,
                                search_settings={"include_domains": ["*.edu", "*.gov", "*.org"], "exclude_domains": ["*.com"]}
                            )
                            content = await self.clean_response(response.content)
                            if hasattr(response, 'choices') and response.choices and hasattr(response.choices[0], 'message') and hasattr(response.choices[0].message, 'executed_tools'):
                                for tool in response.choices[0].message.executed_tools:
                                    if tool.get('type') == 'web_search':
                                        web_sources.extend([f"Source: {result['url']}" for result in tool.get('results', [])])
                            content = f"**Web Search Performed**: Fetched recent data using Tavily.\n\n{content}"
                        else:
                            print("Debug: Invoking Groq without web search")
                            response = client.invoke(prompt_messages)
                            content = await self.clean_response(response.content) if model in ['deepseek', 'qwen', 'compound-beta-mini'] else response.content

                        if not content.strip():
                            print(f"Debug: {model} returned empty content, falling back to mock")
                            return self.get_mock_response(user_message, report_focus)
                        sources = [f"Groq {self.llm_models[model]['name']} API Response"] + (
                            web_sources if use_web_search else [f"{name} FAISS Data" for name in report_names]
                        )
                        print(f"Debug: Response received from {model}: {content[:100]}...")
                        return content, sources
                    except Exception as e:
                        print(f"Debug: {model} error: {str(e)}")
                        return self.get_mock_response(user_message, report_focus)
                else:
                    print(f"Debug: No client for model: {model}, falling back to mock")
                    return self.get_mock_response(user_message, report_focus)
            else:
                print(f"Debug: Unsupported model: {model}, falling back to mock")
                return self.get_mock_response(user_message, report_focus)
        except Exception as e:
            print(f"Debug: FAISS or API Error: {str(e)}, falling back to mock mode")
            return self.get_mock_response(user_message, report_focus)

    async def process_message(self, message: str, history: List[Dict], model: str, report_focus: str):
        print(f"Debug: Processing message: {message}, model: {model}, report_focus: {report_focus}")
        if not message.strip():
            print("Debug: Empty message, returning history")
            return history, ""
        self.conversation_history = history
        self.conversation_history.append({"role": "user", "content": message})
        try:
            content, sources = await self.call_llm_api(self.conversation_history, model, report_focus)
            formatted_response = self.format_response(content, sources, report_focus)
            self.conversation_history.append({"role": "assistant", "content": formatted_response})
            self.save_conversation_history()
            print(f"Debug: Successfully processed message, response: {formatted_response[:100]}...")
        except Exception as e:
            error_response = f"<p>⚠️ Error: {html.escape(str(e))}.</p><p>Please try again.</p>"
            print(f"Debug: Process error: {error_response}")
            self.conversation_history.append({"role": "assistant", "content": error_response})
            self.save_conversation_history()
        return self.conversation_history, ""

# Initialize Flask app
app = Flask(__name__)
agent = IPCCLLMAgent()

# Flask API endpoints
@app.route('/process_message', methods=['POST'])
async def process_message():
    try:
        data = request.json
        print(f"Debug: Received request data: {data}")
        message = data.get('message', '')
        history = data.get('history', [])
        model = data.get('model', 'compound-beta-mini')
        report_focus = data.get('report_focus', 'all')
        history, _ = await agent.process_message(message, history, model, report_focus)
        print("Debug: Sending response with updated history")
        return jsonify({'history': history})
    except Exception as e:
        print(f"Debug: Error in process_message endpoint: {str(e)}")
        return jsonify({'history': history + [{'role': 'assistant', 'content': f"<p>⚠️ Server error: {html.escape(str(e))}.</p><p>Please try again.</p>"}]}), 500

@app.route('/new_session', methods=['POST'])
def new_session():
    try:
        history = agent.new_session()
        print(f"Debug: New session created: {agent.session_id}")
        return jsonify({'history': history, 'session_id': agent.session_id})
    except Exception as e:
        print(f"Debug: Error in new_session: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/switch_session', methods=['POST'])
def switch_session():
    try:
        data = request.json
        session_id = data.get('session_id', agent.session_id)
        history = agent.switch_session(session_id)
        print(f"Debug: Switched to session: {session_id}")
        return jsonify({'history': history, 'session_id': session_id})
    except Exception as e:
        print(f"Debug: Error in switch_session: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/clear_session', methods=['POST'])
def clear_session():
    try:
        history = agent.new_session()
        print(f"Debug: Cleared session, new session: {agent.session_id}")
        return jsonify({'history': history, 'session_id': agent.session_id})
    except Exception as e:
        print(f"Debug: Error in clear_session: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/get_sessions', methods=['GET'])
def get_sessions():
    try:
        sessions = agent.get_session_list()
        print(f"Debug: Retrieved sessions: {sessions}")
        return jsonify({'sessions': sessions})
    except Exception as e:
        print(f"Debug: Error in get_sessions: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route("/chat_ui.html", methods=["GET"])
def serve_chat_ui():
    print("Debug: Serving chat_ui.html from templates/")
    return render_template("chat_ui.html")

# HTML UI
try:
    html_content = f"""
<!DOCTYPE html>
<html lang="en" data-theme="dark">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>AI-Powered Analysis of IPCC Reports</title>
  <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.4/css/all.min.css" integrity="sha512-1ycn6IcaQQ40/MKBW2W4Rhis/DbILU74C1vSrLJxCq57o941Ym01SwNsOMqvEBFlcgUa6xLiPY/NS5R+E6ztJQ==" crossorigin="anonymous" referrerpolicy="no-referrer" />
  <style>
    :root {{
      --bg-dark: #343541;
      --sidebar-dark: #202123;
      --text-dark: #ffffff;
      --input-dark: #40414f;
      --user-msg: #000000;
      --bot-msg: #444654;
      --highlight: #10a37f;
      --bg-light: #f0f0f0;
      --sidebar-light: #ffffff;
      --text-light: #111111;
      --input-light: #dddddd;
      --user-msg-light: #D3D3D3;
      --bot-msg-light: #cfcfcf;
    }}

    [data-theme="dark"] {{
      --bg: var(--bg-dark);
      --sidebar: var(--sidebar-dark);
      --text: var(--text-dark);
      --input: var(--input-dark);
      --user-msg: var(--sidebar-dark);
      --bot-msg: var(--bot-msg);
    }}

    [data-theme="light"] {{
      --bg: var(--bg-light);
      --sidebar: var(--sidebar-light);
      --text: var(--text-light);
      --input: var(--input-light);
      --user-msg: var(--user-msg-light);
      --bot-msg: var(--bot-msg-light);
    }}

    [data-theme="light"] .sidebar {{
      border-right: 1px solid #ffffff;
    }}

    [data-theme="light"] .main-header {{
      border-bottom: 1px solid #ffffff;
    }}

    [data-theme="light"] .input-bar {{
      border-top: 1px solid #ffffff;
    }}

    * {{
      box-sizing: border-box;
    }}

    body {{
      margin: 0;
      font-family: Arial, sans-serif;
      display: flex;
      height: 100vh;
      background-color: var(--bg);
      color: var(--text);
      transition: background-color 0.3s, color 0.3s;
    }}

    .sidebar {{
      width: 300px;
      background-color: var(--sidebar);
      padding: 1rem;
      display: flex;
      flex-direction: column;
      border-right: 1px solid #2d2e30;
      overflow-y: auto;
      transition: width 0.3s ease, transform 0.3s ease;
      z-index: 1000;
    }}

    .sidebar.collapsed {{
      width: 0;
      transform: translateX(-100%);
      padding: 0;
      overflow: hidden;
    }}

    .sidebar.collapsed .topbar,
    .sidebar.collapsed .quick-prompts,
    .sidebar.collapsed h2,
    .sidebar.collapsed h3,
    .sidebar.collapsed #chatHistory {{
      display: none;
    }}

    .sidebar h2, .topbar h3 {{
      margin: 0;
      font-size: 16px;
      color: var(--text);
      margin-bottom: 0;
    }}

    .toggle-sidebar {{
      background: var(--bg);
      color: white;
      border: none;
      border-radius: 5px;
      padding: 0.5rem;
      cursor: pointer;
      font-weight: bold;
      font-size: 16px;
      display: block;
    }}

    .toggle-sidebar:hover {{
      background: var(--user-msg);
    }}

    .topbar {{
      display: flex;
      flex-direction: column;
      gap: 10px;
      margin-bottom: 1rem;
    }}

    .topbar button, #chatHistory select, .quick-prompts button {{
      padding: 0.5rem;
      font-size: 14px;
      border-radius: 5px;
      border: none;
      background: var(--sidebar);
      color: var(--text);
      cursor: pointer;
      width: 100%;
      text-align: left;
    }}

    .topbar button:hover, #chatHistory select:hover, .quick-prompts button:hover {{
      background: var(--input);
    }}

    .quick-prompts {{
      margin-top: 10px;
    }}

    .quick-prompts button {{
      padding: 8px;
    }}

    .main {{
      flex: 1;
      display: flex;
      flex-direction: column;
      justify-content: space-between;
      transition: margin-left 0.3s ease;
    }}

    .main-header {{
      background: var(--bg);
      color: var(--text);
      text-align: center;
      padding: 20px;
      border-bottom: 1px solid #2d2e30;
    }}

    .main-header h1 {{
      margin: 0;
      font-size: 24px;
    }}

    .chat-window {{
      padding: 1rem;
      overflow-y: auto;
      flex-grow: 1;
    }}

    .message {{
      max-width: 95%;
      padding: 1rem;
      margin: 0.5rem 0;
      border-radius: 10px;
      white-space: pre-wrap;
    }}

    .message.user {{
      align-self: flex-end;
      background-color: var(--user-msg);
      text-align: right;
      margin-left: auto;
      margin-right: 10px;
    }}

    .message.bot {{
      align-self: flex-start;
      background-color: var(--bot-msg);
      text-align: left;
      margin-left: 10px;
      margin-right: auto;
    }}

    .message.bot .response-content p {{
      margin: 0.5rem 0;
    }}

    .message.bot .response-content ul {{
      margin: 0.5rem 0;
      padding-left: 20px;
    }}

    .message.bot .response-content .sources-list {{
      margin-top: 1rem;
      font-size: 0.9em;
      color: var(--text);
    }}

    .input-bar {{
      padding: 1rem;
      border-top: 1px solid #2d2e30;
      background-color: var(--bg);
    }}

    .input-container {{
      display: flex;
      flex-direction: column;
      background-color: var(--input);
      padding: 0.75rem;
      border-radius: 10px;
      position: relative;
    }}

    .input-container textarea {{
      flex: 1;
      background: transparent;
      border: none;
      color: var(--text);
      font-size: 16px;
      outline: none;
      resize: vertical;
      min-height: 40px;
      max-height: 150px;
      overflow-y: auto;
      line-height: 1.5;
      width: 100%;
      margin-bottom: 10px;
    }}

    .input-container .button-container {{
      display: flex;
      justify-content: space-between;
      align-items: center;
    }}

    .input-container button {{
      background: #000000;
      border: none;
      color: #ffffff;
      padding: 0.5rem 1rem;
      border-radius: 8px;
      cursor: pointer;
      font-weight: bold;
    }}

    .input-container button:hover {{
      background: #1a1a1a;
    }}

    .input-container .tools-button {{
      display: flex;
      align-items: center;
      gap: 0.5rem;
      font-size: 14px;
      padding: 0.5rem;
    }}

    .tools-menu {{
      position: absolute;
      bottom: 100%;
      left: 0;
      background-color: var(--sidebar);
      border: 1px solid #2d2e30;
      border-radius: 5px;
      z-index: 1000;
      width: 250px;
      max-height: 300px;
      overflow-y: auto;
      display: none;
      box-shadow: 0 2px 5px rgba(0,0,0,0.3);
    }}

    .tools-menu.active {{
      display: block;
    }}

    .tools-menu .menu-section {{
      padding: 0.5rem;
    }}

    .tools-menu .menu-section h3 {{
      margin: 0;
      font-size: 14px;
      color: var(--text);
      padding: 0.5rem;
      border-bottom: 1px solid #2d2e30;
      cursor: pointer;
      display: flex;
      justify-content: space-between;
      align-items: center;
    }}

    .tools-menu .menu-section h3::after {{
      content: '▼';
      font-size: 12px;
      transition: transform 0.2s;
    }}

    .tools-menu .menu-section.collapsed h3::after {{
      content: '▶';
    }}

    .tools-menu .menu-items {{
      display: block;
      transition: max-height 0.3s ease, opacity 0.3s ease;
      max-height: 300px;
      opacity: 1;
      overflow: hidden;
    }}

    .tools-menu .menu-section.collapsed .menu-items {{
      max-height: 0;
      opacity: 0;
      overflow: hidden;
    }}

    .tools-menu .menu-item {{
      padding: 0.5rem;
      font-size: 14px;
      color: var(--text);
      cursor: pointer;
      border-radius: 5px;
    }}

    .tools-menu .menu-item:hover {{
      background: var(--input);
    }}

    /* Mobile-specific styles */
    @media (max-width: 768px) {{
      body {{
        flex-direction: column;
      }}

      .sidebar {{
        position: fixed;
        top: 0;
        left: 0;
        height: 100vh;
        width: 80vw;
        max-width: 250px;
        transform: translateX(-100%);
        box-shadow: 2px 0 5px rgba(0,0,0,0.3);
        z-index: 1000;
      }}

      .sidebar.collapsed {{
        transform: translateX(-100%);
        width: 0;
        padding: 0;
      }}

      .sidebar:not(.collapsed) {{
        transform: translateX(0);
      }}

      .sidebar h2, .topbar h3 {{
        font-size: 14px;
      }}

      .toggle-sidebar {{
        position: fixed;
        top: 10px;
        left: 10px;
        width: 40px;
        height: 40px;
        padding: 0;
        font-size: 20px;
        z-index: 1001;
        display: flex;
        align-items: center;
        justify-content: center;
      }}

      .main {{
        margin-left: 0;
        width: 100%;
      }}

      .main-header h1 {{
        font-size: 20px;
      }}

      .chat-window {{
        padding: 0.5rem;
      }}

      .message {{
        max-width: 97%;
        padding: 0.75rem;
        font-size: 14px;
      }}

      .message.user {{
        margin-right: 5px;
      }}

      .message.bot {{
        margin-left: 5px;
      }}

      .message.bot .response-content p {{
        margin: 0.4rem 0;
      }}

      .message.bot .response-content ul {{
        margin: 0.4rem 0;
        padding-left: 15px;
      }}

      .message.bot .response-content .sources-list {{
        font-size: 0.8em;
      }}

      .input-bar {{
        padding: 0.5rem;
      }}

      .input-container textarea {{
        font-size: 14px;
        min-height: 36px;
      }}

      .input-container button {{
        padding: 0.5rem;
        font-size: 14px;
      }}

      .input-container .tools-button {{
        font-size: 12px;
        padding: 0.4rem;
      }}

      .tools-menu {{
        width: 200px;
        max-height: 200px;
      }}

      .tools-menu .menu-section h3 {{
        font-size: 12px;
      }}

      .tools-menu .menu-item {{
        font-size: 12px;
        padding: 0.4rem;
      }}
    }}

    @media (max-width: 480px) {{
      .sidebar h2, .topbar h3 {{
        font-size: 12px;
      }}

      .main-header h1 {{
        font-size: 18px;
      }}

      .message {{
        max-width: 98%;
        padding: 0.5rem;
        font-size: 12px;
      }}

      .message.user {{
        margin-right: 3px;
      }}

      .message.bot {{
        margin-left: 3px;
      }}

      .message.bot .response-content p {{
        margin: 0.3rem 0;
      }}

      .message.bot .response-content ul {{
        margin: 0.3rem 0;
        padding-left: 10px;
      }}

      .message.bot .response-content .sources-list {{
        font-size: 0.7em;
      }}

      .input-container textarea {{
        font-size: 12px;
        min-height: 32px;
      }}

      .input-container button {{
        padding: 0.4rem;
        font-size: 12px;
      }}

      .input-container .tools-button {{
        font-size: 10px;
        padding: 0.3rem;
      }}

      .tools-menu {{
        width: 180px;
        max-height: 180px;
      }}

      .tools-menu .menu-section h3 {{
        font-size: 10px;
      }}

      .tools-menu .menu-item {{
        font-size: 10px;
        padding: 0.3rem;
      }}
    }}
  </style>
</head>
<body>
  <button class="toggle-sidebar" id="toggleSidebar">☰</button>
  <div class="sidebar" id="sidebar">
    <div class="topbar">
      <h3>Controls</h3>
      <button id="themeToggle">Toggle Theme 🌓</button>
      <button id="newSession">New Session</button>
      <button id="clearSession">Clear Current Session</button>
    </div>
    <h2>Quick Prompts</h2>
    <div class="quick-prompts">
      <button onclick="setPrompt('Summarize SROCC key findings')">Summarize SROCC key findings</button>
      <button onclick="setPrompt('Key points from SR15 1.5°C report')">Key points from SR15 1.5°C report</button>
      <button onclick="setPrompt('Land use impacts from SRCCL')">Land use impacts from SRCCL</button>
      <button onclick="setPrompt('Summarize AR6 WGIII mitigation strategies')">Summarize AR6 WGIII mitigation strategies</button>
      <button onclick="setPrompt('Explain AR6 WGII impacts')">Explain AR6 WGII impacts</button>
    </div>
    <h2>History</h2>
    <div id="chatHistory">
      <select id="sessionSelector"></select>
    </div>
  </div>

  <div class="main" id="main">
    <div class="main-header">
      <h1>AI-Powered Analysis of IPCC Reports</h1>
    </div>
    <div class="chat-window" id="chatWindow">
      <div class="message bot"><div class="response-content"><p>Hello! I'm your IPCC LLM assistant. Use the Tools menu to select a model and report focus, or use a quick prompt to get started.</p></div></div>
    </div>
    <div class="input-bar">
      <div class="input-container">
        <textarea id="userInput" placeholder="Ask about IPCC reports or search recent data..."></textarea>
        <div class="button-container">
          <button class="tools-button" id="toolsButton"><i class="fas fa-sliders-h"></i> Tools</button>
          <div class="tools-menu" id="toolsMenu">
            <div class="menu-section" id="modelSection">
              <h3 onclick="toggleSection('modelSection')">AI Model</h3>
              <div class="menu-items">
                <div class="menu-item" onclick="selectModel('compound-beta-mini')">Compound-Beta-Mini</div>
                <div class="menu-item" onclick="selectModel('deepseek')">DeepSeek-R1-Distill-Llama-70B</div>
                <div class="menu-item" onclick="selectModel('llama')">Llama-3.3-70B-Versatile</div>
                <div class="menu-item" onclick="selectModel('gemma2')">Gemma2-9B-IT</div>
                <div class="menu-item" onclick="selectModel('qwen')">Qwen3-32B</div>
                <div class="menu-item" onclick="selectModel('mock')">Mock AI (Demo)</div>
              </div>
            </div>
            <div class="menu-section" id="reportSection">
              <h3 onclick="toggleSection('reportSection')">Report Focus</h3>
              <div class="menu-items">
                <div class="menu-item" onclick="selectReport('all')">All IPCC Reports</div>
                <div class="menu-item" onclick="selectReport('srocc')">SROCC Summary for Policymakers (2019)</div>
                <div class="menu-item" onclick="selectReport('ar6_syr_full')">AR6 Synthesis Report Full Volume (2023)</div>
                <div class="menu-item" onclick="selectReport('ar6_syr_slides')">AR6 Synthesis Report Slide Deck (2023)</div>
                <div class="menu-item" onclick="selectReport('ar6_wgii_ts')">AR6 WGII Technical Summary (2022)</div>
                <div class="menu-item" onclick="selectReport('ar6_wgiii')">AR6 WGIII Full Report (2022)</div>
                <div class="menu-item" onclick="selectReport('sr15')">SR15 1.5°C Full Report (2018)</div>
                <div class="menu-item" onclick="selectReport('srccl')">SRCCL Full Report (2019)</div>
              </div>
            </div>
          </div>
          <button id="sendButton">Send</button>
        </div>
      </div>
    </div>
  </div>

  <script>
    let currentSessionId = "{agent.session_id}";
    let history = {json.dumps(agent.conversation_history)};
    let currentModel = "compound-beta-mini";
    let currentReport = "all";

    function updateChatWindow() {{
      const chatWindow = document.getElementById("chatWindow");
      chatWindow.innerHTML = "";
      history.forEach(msg => {{
        const div = document.createElement("div");
        div.className = 'message ' + msg.role;
        div.innerHTML = msg.role === "user" ? msg.content : msg.content;
        chatWindow.appendChild(div);
      }});
      chatWindow.scrollTop = chatWindow.scrollHeight;
    }}

    function updateSessionList() {{
      fetch('/get_sessions')
        .then(response => response.json())
        .then(data => {{
          const sessionSelector = document.getElementById("sessionSelector");
          sessionSelector.innerHTML = "";
          data.sessions.forEach(session => {{
            const option = document.createElement("option");
            option.value = session;
            option.textContent = session;
            if (session === currentSessionId) option.selected = true;
            sessionSelector.appendChild(option);
          }});
        }})
        .catch(error => console.error("Error fetching sessions:", error));
    }}

    function sendMessage() {{
      const input = document.getElementById("userInput");
      if (!input.value.trim()) return;

      fetch('/process_message', {{
        method: 'POST',
        headers: {{ 'Content-Type': 'application/json' }},
        body: JSON.stringify({{ message: input.value, history: history, model: currentModel, report_focus: currentReport }})
      }})
        .then(response => {{
          console.log("Response status:", response.status);
          if (!response.ok) throw new Error('Network response was not ok: ' + response.status);
          return response.json();
        }})
        .then(data => {{
          console.log("Received response:", data);
          history = data.history;
          updateChatWindow();
          updateSessionList();
          input.style.height = 'auto';
        }})
        .catch(error => {{
          console.error("Error processing message:", error);
          history.push({{ role: "assistant", content: `<p>⚠️ Error processing message: ${{error.message}}.</p><p>Please try again.</p>` }});
          updateChatWindow();
          input.style.height = 'auto';
        }});

      input.value = "";
    }}

    function autoResizeTextarea() {{
      const textarea = document.getElementById("userInput");
      textarea.style.height = 'auto';
      textarea.style.height = `${{Math.min(textarea.scrollHeight, 150)}}px`;
    }}

    function handleKeyPress(event) {{
      const textarea = document.getElementById("userInput");
      if (event.key === "Enter" && !event.shiftKey) {{
        event.preventDefault();
        sendMessage();
      }}
    }}

    function setPrompt(prompt) {{
      const input = document.getElementById("userInput");
      input.value = prompt;
      autoResizeTextarea();
      sendMessage();
    }}

    function toggleTheme() {{
      try {{
        const current = document.documentElement.getAttribute("data-theme");
        document.documentElement.setAttribute("data-theme", current === "dark" ? "light" : "dark");
      }} catch (e) {{
        console.error("Error in toggleTheme:", e);
      }}
    }}

    function toggleSidebar() {{
      const sidebar = document.getElementById("sidebar");
      const toggleButton = document.getElementById("toggleSidebar");
      sidebar.classList.toggle("collapsed");
      toggleButton.textContent = sidebar.classList.contains("collapsed") ? "☰" : "✕";
    }}

    function closeSidebarOnMainClick(event) {{
      const sidebar = document.getElementById("sidebar");
      const toggleButton = document.getElementById("toggleSidebar");
      if (!sidebar.classList.contains("collapsed") &&
          !sidebar.contains(event.target) &&
          !toggleButton.contains(event.target) &&
          window.innerWidth <= 768) {{
        sidebar.classList.add("collapsed");
        toggleButton.textContent = "☰";
      }}
    }}

    function initializeSidebar() {{
      const sidebar = document.getElementById("sidebar");
      const toggleButton = document.getElementById("toggleSidebar");
      if (window.innerWidth <= 768) {{
        sidebar.classList.add("collapsed");
        toggleButton.textContent = "☰";
      }} else {{
        sidebar.classList.remove("collapsed");
        toggleButton.textContent = "✕";
      }}
    }}

    function newSession() {{
      fetch('/new_session', {{
        method: 'POST',
        headers: {{ 'Content-Type': 'application/json' }}
      }})
        .then(response => response.json())
        .then(data => {{
          history = data.history;
          currentSessionId = data.session_id;
          updateChatWindow();
          updateSessionList();
        }})
        .catch(error => console.error("Error creating new session:", error));
    }}

    function switchSession() {{
      const sessionId = document.getElementById("sessionSelector").value;
      fetch('/switch_session', {{
        method: 'POST',
        headers: {{ 'Content-Type': 'application/json' }},
        body: JSON.stringify({{ session_id: sessionId }})
      }})
        .then(response => response.json())
        .then(data => {{
          history = data.history;
          currentSessionId = data.session_id;
          updateChatWindow();
          updateSessionList();
        }})
        .catch(error => console.error("Error switching session:", error));
    }}

    function clearSession() {{
      fetch('/clear_session', {{
        method: 'POST',
        headers: {{ 'Content-Type': 'application/json' }}
      }})
        .then(response => response.json())
        .then(data => {{
          history = data.history;
          currentSessionId = data.session_id;
          updateChatWindow();
          updateSessionList();
        }})
        .catch(error => console.error("Error clearing session:", error));
    }}

    function toggleToolsMenu() {{
      const toolsMenu = document.getElementById("toolsMenu");
      toolsMenu.classList.toggle("active");
    }}

    function selectModel(model) {{
      currentModel = model;
      document.getElementById("toolsMenu").classList.remove("active");
    }}

    function selectReport(report) {{
      currentReport = report;
      document.getElementById("toolsMenu").classList.remove("active");
    }}

    function toggleSection(sectionId) {{
      const section = document.getElementById(sectionId);
      section.classList.toggle("collapsed");
    }}

    function closeToolsMenu(event) {{
      const toolsMenu = document.getElementById("toolsMenu");
      const toolsButton = document.getElementById("toolsButton");
      if (!toolsMenu.contains(event.target) && !toolsButton.contains(event.target)) {{
        toolsMenu.classList.remove("active");
      }}
    }}

    document.addEventListener('DOMContentLoaded', () => {{
      document.getElementById('sendButton').addEventListener('click', sendMessage);
      document.getElementById('themeToggle').addEventListener('click', toggleTheme);
      document.getElementById('toggleSidebar').addEventListener('click', toggleSidebar);
      document.getElementById('newSession').addEventListener('click', newSession);
      document.getElementById('clearSession').addEventListener('click', clearSession);
      document.getElementById('sessionSelector').addEventListener('change', switchSession);
      document.getElementById('main').addEventListener('click', closeSidebarOnMainClick);
      document.getElementById('toolsButton').addEventListener('click', toggleToolsMenu);
      document.addEventListener('click', closeToolsMenu);
      const textarea = document.getElementById("userInput");
      textarea.addEventListener('input', autoResizeTextarea);
      textarea.addEventListener('keypress', handleKeyPress);
      updateChatWindow();
      updateSessionList();
      initializeSidebar();
      window.addEventListener('resize', initializeSidebar);
    }});
  </script>
</body>
</html>
"""
except Exception as e:
    print(f"Error formatting HTML content: {str(e)}")
    raise

# Save HTML to a file
with open("chat_ui.html", "w") as f:
    f.write(html_content)

# ---------------- Local Environment Fix ----------------
# Skip Google Drive mounting when running outside Colab
if os.getenv("COLAB_ENV", "0") == "1":
    try:
        mount_point = "/content/drive"
        os.makedirs(mount_point, exist_ok=True)
    except Exception as e:
        print(f"Error mounting Google Drive: {e}")
        raise RuntimeError("Unable to mount Google Drive. Please check access or restart runtime.")
else:
    print("Running locally, skipping Google Drive mount.")
   

# Retrieve ngrok authtoken
try:
    # NGROK_AUTH_TOKEN = userdata.get('NGROK_AUTH_TOKEN')
    # NGROK_AUTH_TOKEN = "307H4NSfyL4M1mup2b4wPpgPVAQ_4MkXhGNqBEU9EuNViRG3G"  # e.g., "2ABCdefGHIjkl45mnopQRSTuv6789"
    # if not NGROK_AUTH_TOKEN:
    #     raise ValueError("NGROK_AUTH_TOKEN not found in Colab Secrets")
    # if NGROK_AUTH_TOKEN.startswith('ak_'):
    #     raise ValueError("NGROK_AUTH_TOKEN appears to be an API key, not a tunnel authtoken")
    # ngrok.set_auth_token(NGROK_AUTH_TOKEN)

    print("Running locally, skipping ngrok setup.")

except Exception as e:
    print(f"Error retrieving NGROK_AUTH_TOKEN: {e}")
    print("Set NGROK_AUTH_TOKEN in Colab Secrets (key icon). "
          "Get tunnel authtoken (starts with '2' or '1') from https://dashboard.ngrok.com/get-started/your-authtoken")
    raise


# Find an available port
def find_free_port(start_port=8080, max_attempts=10):
    port = start_port
    for _ in range(max_attempts):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(('localhost', port)) != 0:
                return port
            port += 1
    raise Exception("No available ports found")

PORT = find_free_port()
print(f"Selected port: {PORT}")

# Check and free port if in use
if is_port_in_use := lambda port: socket.socket(socket.AF_INET, socket.SOCK_STREAM).connect_ex(('localhost', port)) == 0:
    if is_port_in_use(PORT):
        print(f"Port {PORT} is in use. Attempting to free it...")
        subprocess.run(["fuser", "-k", f"{PORT}/tcp"])
        time.sleep(1)
        if is_port_in_use(PORT):
            print(f"Error: Port {PORT} is still in use. Trying next port...")
            PORT = find_free_port(PORT + 1)
            print(f"Selected new port: {PORT}")

# Start Flask server in a background thread
def start_flask():
    try:
        app.run(host="0.0.0.0", port=PORT, debug=False, use_reloader=False)
    except Exception as e:
        print(f"Flask server error: {e}")
        raise

flask_thread = threading.Thread(target=start_flask, daemon=True)
flask_thread.start()

# Wait for Flask to start
time.sleep(2)

# Set up ngrok tunnel
try:
    ngrok.kill()
    public_url = ngrok.connect(PORT, bind_tls=True).public_url
    print(f"Access your HTML UI at: {public_url}/chat_ui.html")
    print("Keep this cell running to maintain the server. Stop the cell to terminate.")
except Exception as e:
    print(f"Error setting up ngrok tunnel: {e}")
    print("Possible causes:")
    print("- Invalid NGROK_AUTH_TOKEN: Ensure you used the tunnel authtoken (starts with '2' or '1')")
    print("- Ngrok service issue: Check your ngrok dashboard for account status or limits.")
    print("- Network issues: Ensure Colab has internet access.")
    raise

# Keep runtime alive
try:
    while True:
        time.sleep(60)
except KeyboardInterrupt:
    print("Shutting down Flask server and ngrok tunnel...")
    ngrok.kill()
    print("Server and tunnel terminated.")

if __name__ == "__main__":
    from sentence_transformers import SentenceTransformer

    # Load FAISS index and documents
    from build_faiss import FAISS_BASE_PATH
    import faiss, pickle, os

    def load_faiss_index(name="demo"):
        index_path = os.path.join(FAISS_BASE_PATH, f"{name}.index")
        docs_path = os.path.join(FAISS_BASE_PATH, f"{name}.pkl")

        if not os.path.exists(index_path) or not os.path.exists(docs_path):
            print("❌ FAISS index or documents not found. Run build_faiss.py first.")
            return None, None

        index = faiss.read_index(index_path)
        with open(docs_path, "rb") as f:
            docs = pickle.load(f)
        return index, docs

    # Test retrieval
    index, docs = load_faiss_index("demo")
    if index:
        query = "What is FAISS used for?"
        model = SentenceTransformer("all-MiniLM-L6-v2")
        q_emb = model.encode([query])

        D, I = index.search(q_emb, k=2)
        for idx in I[0]:
            print("🔎 Retrieved:", docs[idx])
