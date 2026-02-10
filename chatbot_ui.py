import tkinter as tk
from tkinter import ttk, scrolledtext, filedialog, messagebox, simpledialog
from datetime import datetime
import threading
from queue import Queue
import sys
from typing import Optional, Callable, Dict, List, Tuple
import logging
import os
from ttkbootstrap import Style  # You'll need to: pip install ttkbootstrap
import re
from afile import LLaMAAPI, Config, setup_logging
from database_manager import DatabaseManager

class ProjectStats:
    def __init__(self):
        self.message_count: int = 0
        self.last_used: Optional[datetime] = None
        self.description: str = ""
        self.tags: List[str] = []

class CustomScrollbar(ttk.Frame):
    """Custom scrollbar widget with modern styling."""
    def __init__(self, parent, *args, **kwargs):
        super().__init__(parent)
        self.style = ttk.Style()
        
        # Configure scrollbar style
        self.style.configure("Custom.Vertical.TScrollbar",
                            background="#404040",
                            troughcolor="#262626",
                            width=10,
                            arrowcolor="#808080")
        
        # Create scrollbar
        self.scrollbar = ttk.Scrollbar(self, style="Custom.Vertical.TScrollbar", *args, **kwargs)
        self.scrollbar.pack(fill="y", expand=True)

class ChatbotUI:
    def __init__(self, llama_api: LLaMAAPI, db_manager: DatabaseManager):
        # Initialize main window with ttkbootstrap style
        self.style = Style(theme="darkly")
        self.root = self.style.master
        self.root.title("Tre's AI Assistant")
        self.root.geometry("1000x800")

        self.llama_api = llama_api  # Store the LLaMAAPI instance
        self.ask_question_callback = llama_api.ask_question
        self.db_manager = db_manager
    # MODEL_NAME: str = "llama3.3:latest"
    # MODEL_NAME: str = "qwen2.5:14b"
#     hf.co/mradermacher/DeepSeek-R1-Distill-Qwen-32B-abliterated-i1-GGUF:Q6_K    405c6413d389    26 GB     3 hours ago    
# deepseek-r1:8b                                                              28f8fd6cdc67    4.9 GB    4 hours ago    
# deepseek-r1:1.5b                                                            a42b25d8c10a    1.1 GB    2 days ago     
# deepseek-r1:32b                                                             38056bbcbb2d    19 GB     2 days ago     
# deepseek-r1:70b 
        self.models = {
            "r1 32b": "deepseek-r1:32b",
            "r1 1.5b": "deepseek-r1:1.5b",
            "r1 32b": "deepseek-r1:32b",
            "r1 70b": "deepseek-r1:70b",
            "abliterated": "hf.co/mradermacher/DeepSeek-R1-Distill-Qwen-32B-abliterated-i1-GGUF:Q6_K"
        }
        
        # Create header frame for model selection and project selection
        header_frame = ttk.Frame(self.root, style="Dark.TFrame")
        header_frame.pack(fill="x", padx=10, pady=(10, 0))
        
        # Create model selection frame
        model_frame = ttk.Frame(header_frame, style="Dark.TFrame")
        model_frame.pack(side="left", fill="x", expand=True)
        
        # Model selector label
        model_label = ttk.Label(
            model_frame,
            text="Model:",
            style="Dark.TLabel"
        )
        model_label.pack(side="left", padx=(0, 5))
        
        # Model selector combobox
        self.model_var = tk.StringVar()
        self.model_combo = ttk.Combobox(
            model_frame,
            textvariable=self.model_var,
            values=list(self.models.keys()),
            state="readonly",
            width=20
        )
        self.model_combo.pack(side="left", padx=(0, 10))
        
        # Set default model
        self.model_combo.set(next(iter(self.models.keys())))  # Set first model as default
        
        # Bind model selection change
        self.model_combo.bind('<<ComboboxSelected>>', self._on_model_selected)
        
        # Configure colors
        self.colors = {
            "bg_dark": "#171717",
            "bg_darker": "#0f0f0f",
            "bg_lighter": "#262626",
            "text_primary": "#FFFFFF",
            "text_secondary": "#A8A8A8",
            "accent_blue": "#3498db",
            "accent_green": "#2ecc71",
            "accent_red": "#e74c3c",
            "user_message": "#8CB4FF",
            "assistant_message": "#9FF8B5",
            "input_bg": "#262626",
            "button_bg": "#2A2A2A",
            "button_hover": "#404040",
            "border": "#404040"
        }
        
        # Configure styles
        self._configure_styles()
        
        # Create main container
        self.main_container = ttk.Frame(self.root, style="Dark.TFrame")
        self.main_container.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Create project management section
        self._create_project_section()
        
        # Create chat section
        self._create_chat_section()
        
        # Initialize other variables
        self.message_queue = Queue()
        self.processing = False
        
        # Start message processing thread
        self.process_thread = threading.Thread(target=self._process_messages, daemon=True)
        self.process_thread.start()
        
        # Update project list
        self._update_project_list()

    def _configure_styles(self):
        """Configure ttk styles for widgets."""
        self.style.configure("Dark.TFrame", background=self.colors["bg_dark"])
        self.style.configure("Dark.TLabel", 
                           background=self.colors["bg_dark"],
                           foreground=self.colors["text_primary"])
        self.style.configure("Dark.TButton",
                           background=self.colors["button_bg"],
                           foreground=self.colors["text_primary"],
                           padding=5)
        self.style.map("Dark.TButton",
                      background=[("active", self.colors["button_hover"])])
        self.style.configure("Dark.TEntry",
                           fieldbackground=self.colors["input_bg"],
                           foreground=self.colors["text_primary"])
        self.style.configure("Project.TFrame",
                           background=self.colors["bg_lighter"],
                           relief="solid",
                           borderwidth=1)

    def _create_project_section(self):
        """Create the project management section."""
        # Project frame
        project_frame = ttk.Frame(self.main_container, style="Dark.TFrame")
        project_frame.pack(fill="x", pady=(0, 10))
        
        # Project selection frame
        selection_frame = ttk.Frame(project_frame, style="Dark.TFrame")
        selection_frame.pack(fill="x", pady=(0, 5))
        
        # Project label and combo
        ttk.Label(selection_frame, text="Project:", style="Dark.TLabel").pack(side="left", padx=(0, 5))
        self.project_var = tk.StringVar()
        self.project_combo = ttk.Combobox(
            selection_frame,
            textvariable=self.project_var,
            state="readonly",
            width=30
        )
        self.project_combo.pack(side="left", padx=(0, 5))
        
        # Project buttons
        ttk.Button(selection_frame, text="New", style="Dark.TButton",
                  command=self._create_new_project).pack(side="left", padx=2)
        ttk.Button(selection_frame, text="Delete", style="Dark.TButton",
                  command=self._delete_project).pack(side="left", padx=2)
        ttk.Button(selection_frame, text="Rename", style="Dark.TButton",
                  command=self._rename_project).pack(side="left", padx=2)
        ttk.Button(selection_frame, text="Export", style="Dark.TButton",
                  command=self._export_project).pack(side="left", padx=2)
        ttk.Button(selection_frame, text="Import", style="Dark.TButton",
                  command=self._import_project).pack(side="left", padx=2)
        
        # Project info frame
        info_frame = ttk.Frame(project_frame, style="Project.TFrame")
        info_frame.pack(fill="x", pady=5)
        
        # Project stats
        self.stats_label = ttk.Label(info_frame, style="Dark.TLabel",
                                   text="Messages: 0 | Last used: Never")
        self.stats_label.pack(side="left", padx=5, pady=5)
        
        # Project description
        self.description_var = tk.StringVar()
        description_entry = ttk.Entry(info_frame, textvariable=self.description_var,
                                    style="Dark.TEntry", width=50)
        description_entry.pack(side="left", padx=5, pady=5, fill="x", expand=True)
        description_entry.bind('<FocusOut>', self._update_description)
        
        # Project tags
        self.tags_var = tk.StringVar()
        tags_entry = ttk.Entry(info_frame, textvariable=self.tags_var,
                             style="Dark.TEntry", width=30)
        tags_entry.pack(side="left", padx=5, pady=5)
        tags_entry.bind('<FocusOut>', self._update_tags)
        ttk.Label(info_frame, text="Tags (comma-separated)", 
                 style="Dark.TLabel").pack(side="left", padx=(0, 5))
        
        prompt_frame = ttk.Frame(project_frame, style="Project.TFrame")
        prompt_frame.pack(fill="x", pady=5)
        
        ttk.Label(prompt_frame, text="System Prompt:", 
                style="Dark.TLabel").pack(side="left", padx=5, pady=5)
        
        self.system_prompt = scrolledtext.ScrolledText(
            prompt_frame,
            height=3,
            wrap=tk.WORD,
            font=("Arial", 11),
            background=self.colors["input_bg"],
            foreground=self.colors["text_primary"],
            insertbackground=self.colors["text_primary"],
            borderwidth=1,
            highlightthickness=0
        )
        self.system_prompt.pack(side="left", fill="x", expand=True, padx=5, pady=5)
        
        prompt_buttons = ttk.Frame(prompt_frame, style="Dark.TFrame")
        prompt_buttons.pack(side="left", padx=5)
        
        ttk.Button(prompt_buttons, text="Reset Default", 
                style="Dark.TButton",
                command=self._reset_system_prompt).pack(side="top", pady=2)
        
        ttk.Button(prompt_buttons, text="Apply", 
                style="Dark.TButton",
                command=self._apply_system_prompt).pack(side="top", pady=2)
        
        # Set default prompt
        self.system_prompt.insert("1.0", Config.DEFAULT_SYSTEM_PROMPT)

    def _reset_system_prompt(self):
        """Reset system prompt to default."""
        logging.info("Resetting system prompt to default")
        self.system_prompt.delete("1.0", tk.END)
        self.system_prompt.insert("1.0", Config.DEFAULT_SYSTEM_PROMPT)
        self._apply_system_prompt()
        logging.info("System prompt reset completed")

    def _apply_system_prompt(self):
        """Apply the current system prompt."""
        new_prompt = self.system_prompt.get("1.0", tk.END).strip()
        logging.info(f"Applying new system prompt: {new_prompt[:50]}...")
        
        if not new_prompt:
            logging.warning("Empty system prompt detected, using default")
            new_prompt = Config.DEFAULT_SYSTEM_PROMPT
        
        # Update the context manager in the LLaMAAPI instance
        self.llama_api.context_manager.set_system_prompt(new_prompt)
        self._add_system_message("System prompt updated")
        logging.info("System prompt update completed")

    def _create_chat_section(self):
        """Create the chat interface section."""
        # Chat frame
        chat_frame = ttk.Frame(self.main_container, style="Dark.TFrame")
        chat_frame.pack(fill="both", expand=True)
        
        # Toolbar
        toolbar = ttk.Frame(chat_frame, style="Dark.TFrame")
        toolbar.pack(fill="x", pady=(0, 5))
        
        # Search
        self.search_var = tk.StringVar()
        search_entry = ttk.Entry(toolbar, textvariable=self.search_var,
                                style="Dark.TEntry", width=30)
        search_entry.pack(side="left", padx=(0, 5))
        search_entry.bind('<Return>', self._search_messages)
        
        ttk.Button(toolbar, text="Search", style="Dark.TButton",
                command=self._search_messages).pack(side="left", padx=(0, 5))
        
        ttk.Button(toolbar, text="Clear Chat", style="Dark.TButton",
                command=self._clear_chat).pack(side="right")
        
        # Input area
        input_frame = ttk.Frame(chat_frame, style="Dark.TFrame")
        input_frame.pack(fill="x", pady=(5, 0))
        
        # Replace the single file selection with a multi-file frame
        file_frame = ttk.Frame(input_frame, style="Dark.TFrame")
        file_frame.pack(fill="x", pady=(0, 5))
        
        # File list frame
        file_list_frame = ttk.Frame(file_frame, style="Project.TFrame")
        file_list_frame.pack(fill="x", pady=(0, 5))
        
        # File list
        self.file_list = tk.Listbox(
            file_list_frame,
            background=self.colors["input_bg"],
            foreground=self.colors["text_primary"],
            selectmode=tk.MULTIPLE,
            height=3,
            borderwidth=1,
            highlightthickness=0
        )
        self.file_list.pack(side="left", fill="x", expand=True, padx=(5, 0))
        
        # File list scrollbar
        file_scrollbar = ttk.Scrollbar(file_list_frame, orient="vertical")
        file_scrollbar.pack(side="right", fill="y")
        
        # Connect scrollbar
        self.file_list.config(yscrollcommand=file_scrollbar.set)
        file_scrollbar.config(command=self.file_list.yview)
        
        # File buttons frame
        file_buttons_frame = ttk.Frame(file_frame, style="Dark.TFrame")
        file_buttons_frame.pack(fill="x", pady=(0, 5))
        
        # Add file button
        self.add_file_button = ttk.Button(
            file_buttons_frame,
            text="Add Files",
            style="Dark.TButton",
            command=self._add_files
        )
        self.add_file_button.pack(side="left", padx=(0, 5))
        
        # Remove file button
        self.remove_file_button = ttk.Button(
            file_buttons_frame,
            text="Remove Selected",
            style="Dark.TButton",
            command=self._remove_files
        )
        self.remove_file_button.pack(side="left", padx=(0, 5))
        
        # Clear files button
        self.clear_files_button = ttk.Button(
            file_buttons_frame,
            text="Clear All",
            style="Dark.TButton",
            command=self._clear_files
        )
        self.clear_files_button.pack(side="left")
        
        # Store file paths
        self.file_paths = []
        
        # Message input
        self.message_input = scrolledtext.ScrolledText(
            input_frame,
            height=4,  # Adjust or remove this if necessary
            wrap=tk.WORD,
            font=("Arial", 12),
            background=self.colors["input_bg"],
            foreground=self.colors["text_primary"],
            insertbackground=self.colors["text_primary"],
            borderwidth=1,
            highlightthickness=0
        )
        self.message_input.pack(fill="x", pady=5)
        
        # Token counter
        self.token_counter = ttk.Label(input_frame, style="Dark.TLabel",
                                    text="Tokens: 0")
        self.token_counter.pack(side="left")
        
        # Send button
        self.send_button = ttk.Button(
            input_frame,
            text="Send",
            style="Dark.TButton",
            command=self._send_message
        )
        self.send_button.pack(side="right")
        
        # Bind events
        self.message_input.bind("<Control-Return>", lambda e: self._send_message())
        self.message_input.bind("<KeyRelease>", self._update_token_count)
        
        # Chat display
        display_frame = ttk.Frame(chat_frame, style="Dark.TFrame")
        display_frame.pack(fill="both", expand=True)
        
        self.chat_display = scrolledtext.ScrolledText(
            display_frame,
            wrap=tk.WORD,
            font=("Arial", 24),  # Adjust the font size as needed
            # Remove or adjust the height parameter
            # height=20,
            background=self.colors["bg_dark"],
            foreground=self.colors["text_primary"],
            insertbackground=self.colors["text_primary"],
            padx=10,
            pady=10,
            borderwidth=0,
            highlightthickness=0
        )
        self.chat_display.pack(side="left", fill="both", expand=True)
        
        # Custom scrollbar
        scrollbar = CustomScrollbar(display_frame)
        scrollbar.pack(side="right", fill="y")
        self.chat_display.configure(yscrollcommand=scrollbar.scrollbar.set)
        scrollbar.scrollbar.configure(command=self.chat_display.yview)
        
        # Configure text tags
        self.chat_display.tag_configure("user",
            foreground=self.colors["user_message"],
            font=("Arial", 24, "bold")
        )
        self.chat_display.tag_configure("assistant",
            foreground=self.colors["assistant_message"],
            font=("Arial", 24, "bold")
        )
        self.chat_display.tag_configure("system",
            foreground=self.colors["text_secondary"],
            font=("Arial", 24)
        )
        self.chat_display.tag_configure("message",
            foreground=self.colors["text_primary"],
            font=("Arial", 24)
        )

    def _update_token_count(self, event=None):
        """Update the token counter."""
        text = self.message_input.get("1.0", tk.END).strip()
        # You can replace this with your actual token counting function
        token_count = len(text.split())
        self.token_counter.configure(text=f"Tokens: {token_count}")

    def _browse_file(self):
        """Open file browser dialog."""
        filename = filedialog.askopenfilename(
            title="Select a file",
            filetypes=(
                ("Text files", "*.txt"),
                ("Python files", "*.py"),
                ("All files", "*.*")
            )
        )
        if filename:
            self.file_path.set(filename)

    def _process_messages(self):
        """Process messages in the queue."""
        while True:
            message, file_content, model = self.message_queue.get()
            try:
                # Get response
                response = self.llama_api.ask_question(message, file_content)
                
                if response:
                    # Display assistant response
                    self.root.after(0, self._add_assistant_message, response)
                    logging.info(f"Displayed response: {response[:100]}...")  # Log first 100 chars
                else:
                    error_msg = "No response received or response was empty."
                    self.root.after(0, self._add_system_message, error_msg)
                    logging.error(error_msg)
            except Exception as e:
                error_msg = f"Error processing message: {str(e)}"
                self.root.after(0, self._add_system_message, error_msg)
                logging.error(error_msg)
                logging.error("Error details:", exc_info=True)
            finally:
                # Re-enable input
                self.root.after(0, self._toggle_input, True)
                # Clear file list
                self.root.after(0, self._clear_files)

    def _add_user_message(self, message: str):
        """Add a user message to the chat display."""
        self.chat_display.insert(tk.END, "You: ", "user")
        self.chat_display.insert(tk.END, f"{message}\n\n", "message")
        self.chat_display.see(tk.END)

    def _add_assistant_message(self, message: str):
        """Add an assistant message to the chat display."""
        self.chat_display.insert(tk.END, "Assistant: ", "assistant")
        self.chat_display.insert(tk.END, f"{message}\n\n", "message")
        self.chat_display.see(tk.END)

    def _add_system_message(self, message: str):
        """Add a system message to the chat display."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.chat_display.insert(tk.END, f"[{timestamp}] {message}\n", "system")
        self.chat_display.see(tk.END)

    def _add_files(self):
        """Open file browser dialog for multiple files."""
        filenames = filedialog.askopenfilenames(
            title="Select files",
            filetypes=(
                ("Text files", "*.txt"),
                ("Python files", "*.py"),
                ("HTML files", "*.html"),
                ("CSS files", "*.css"),
                ("JavaScript files", "*.js"),
                ("All files", "*.*")
            )
        )
        if filenames:
            for filename in filenames:
                # Check if file exists and is readable
                if os.path.exists(filename) and os.access(filename, os.R_OK):
                    if filename not in self.file_paths:
                        self.file_paths.append(filename)
                        self.file_list.insert(tk.END, os.path.basename(filename))
                        self._add_system_message(f"Added file: {filename}")
                else:
                    self._add_system_message(f"Error: Cannot access file {filename}")

    def _remove_files(self):
        """Remove selected files from the list."""
        selected = self.file_list.curselection()
        for index in reversed(selected):
            filename = self.file_paths[index]
            del self.file_paths[index]
            self.file_list.delete(index)
            self._add_system_message(f"Removed file: {filename}")

    def _clear_files(self):
        """Clear all files from the list."""
        self.file_paths.clear()
        self.file_list.delete(0, tk.END)
        self._add_system_message("Cleared all files")

    def _on_model_selected(self, event=None):
        """Handle model selection change."""
        selected_display_name = self.model_var.get()
        selected_model = self.models[selected_display_name]
        self._add_system_message(f"Switched to model: {selected_display_name}")
        
        # Update the model in the LLaMAAPI instance
        self.llama_api.update_model(selected_model)
        self._add_system_message(f"Model configuration updated successfully")

    def _send_message(self):
        """Handle sending a message."""
        message = self.message_input.get("1.0", tk.END).strip()
        if not message:
            return
            
        if not self.project_var.get():
            self._add_system_message("Please select a project first")
            return
            
        selected_display_name = self.model_var.get()
        selected_model = self.models[selected_display_name]
        
        self.db_manager.set_current_project(self.project_var.get())
        self._toggle_input(False)
        
        # Clear input box - first attempt
        self.message_input.delete("1.0", tk.END)
        self.message_input.update_idletasks()  # Force update
        
        self._add_user_message(message)
        self._add_system_message(f"Using model: {selected_display_name}")
        
        # Clear input box - second attempt
        self.root.after(100, lambda: self.message_input.delete("1.0", tk.END))
        self._update_token_count()
        
        file_contents = {}
        for filepath in self.file_paths:
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
                    file_contents[os.path.basename(filepath)] = content
                    self._add_system_message(f"Successfully read file: {os.path.basename(filepath)}")
            except Exception as e:
                self._add_system_message(f"Error reading file {filepath}: {str(e)}")
        
        if file_contents:
            formatted_content = ""
            for filename, content in file_contents.items():
                formatted_content += f"\n=== File: {filename} ===\n"
                formatted_content += content
                formatted_content += "\n\n"
            
            enhanced_message = (
                "I have the following files to analyze:\n"
                f"{', '.join(file_contents.keys())}\n\n"
                "Here are their contents:\n"
                f"{formatted_content}\n"
                f"Question: {message}"
            )
            self._add_system_message(f"Including {len(file_contents)} files in the request")
            self.message_queue.put((enhanced_message, None, selected_model))
        else:
            self.message_queue.put((message, None, selected_model))

    def _toggle_input(self, enabled: bool):
        """Enable or disable input controls."""
        state = "normal" if enabled else "disabled"
        self.message_input.configure(state=state)
        self.send_button.configure(state=state)
        self.project_combo.configure(state="readonly" if enabled else state)
        self.add_file_button.configure(state=state)
        self.remove_file_button.configure(state=state)
        self.clear_files_button.configure(state=state)
        self.file_list.configure(state=state)

    def _update_project_list(self):
        """Update the project dropdown and stats."""
        projects = self.db_manager.get_projects()
        self.project_combo['values'] = [p[0] for p in projects]
        
        if projects and not self.project_var.get():
            self.project_var.set(projects[0][0])
            self._on_project_selected()
        
        self._update_project_stats()

    def _update_project_stats(self):
        """Update project statistics display."""
        project = self.project_var.get()
        if not project:
            return
        
        projects = dict(self.db_manager.get_projects())
        if project in projects:
            stats = projects[project]
            last_used = stats.last_used.strftime("%Y-%m-%d %H:%M") if stats.last_used else "Never"
            self.stats_label.configure(
                text=f"Messages: {stats.message_count} | Last used: {last_used}"
            )
            self.description_var.set(stats.description)
            self.tags_var.set(", ".join(stats.tags))

    def _on_project_selected(self, event=None):
        """Handle project selection change."""
        project = self.project_var.get()
        if project:
            self.db_manager.set_current_project(project)
            self._add_system_message(f"Switched to project: {project}")
            self._update_project_stats()

    def _create_new_project(self):
        """Create a new project."""
        new_project = simpledialog.askstring(
            "New Project",
            "Enter project name:",
            parent=self.root
        )
        if new_project:
            if not re.match(r'^[a-zA-Z0-9_]+$', new_project):
                messagebox.showerror(
                    "Invalid Name",
                    "Project name can only contain letters, numbers, and underscores"
                )
                return
            
            description = simpledialog.askstring(
                "Project Description",
                "Enter project description (optional):",
                parent=self.root
            )
            
            tags = simpledialog.askstring(
                "Project Tags",
                "Enter project tags (comma-separated, optional):",
                parent=self.root
            )
            tags_list = [t.strip() for t in tags.split(",")] if tags else []
            
            if self.db_manager.create_project(new_project, description, tags_list):
                self._update_project_list()
                self.project_var.set(new_project)
                self._on_project_selected()
            else:
                messagebox.showerror(
                    "Error",
                    "Failed to create new project"
                )

    def _delete_project(self):
        """Delete the current project."""
        project = self.project_var.get()
        if not project:
            return
            
        if messagebox.askyesno(
            "Confirm Delete",
            f"Are you sure you want to delete the project '{project}'?\nThis cannot be undone!"
        ):
            if self.db_manager.delete_project(project):
                self._update_project_list()
                self._add_system_message(f"Deleted project: {project}")
            else:
                messagebox.showerror(
                    "Error",
                    "Failed to delete project"
                )

    def _rename_project(self):
        """Rename the current project."""
        old_name = self.project_var.get()
        if not old_name:
            return
            
        new_name = simpledialog.askstring(
                "Rename Project",
            f"Enter new name for project '{old_name}':",
            parent=self.root
        )
        if new_name:
            if not re.match(r'^[a-zA-Z0-9_]+$', new_name):
                messagebox.showerror(
                    "Invalid Name",
                    "Project name can only contain letters, numbers, and underscores"
                )
                return
                
            if self.db_manager.rename_project(old_name, new_name):
                self._update_project_list()
                self.project_var.set(new_name)
                self._on_project_selected()
            else:
                messagebox.showerror(
                    "Error",
                    "Failed to rename project"
                )

    def _export_project(self):
        """Export the current project to a file."""
        project = self.project_var.get()
        if not project:
            return
            
        filepath = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json")],
            initialfile=f"{project}_export.json"
        )
        if filepath:
            if self.db_manager.export_project(project, filepath):
                self._add_system_message(f"Exported project to: {filepath}")
            else:
                messagebox.showerror(
                    "Error",
                    "Failed to export project"
                )

    def _import_project(self):
        """Import a project from a file."""
        filepath = filedialog.askopenfilename(
            filetypes=[("JSON files", "*.json")]
        )
        if filepath:
            if self.db_manager.import_project(filepath):
                self._update_project_list()
                self._add_system_message(f"Imported project from: {filepath}")
            else:
                messagebox.showerror(
                    "Error",
                    "Failed to import project"
                )

    def _update_description(self, event=None):
        """Update project description."""
        project = self.project_var.get()
        if project:
            description = self.description_var.get()
            if self.db_manager.update_project_metadata(project, description=description):
                self._update_project_stats()

    def _update_tags(self, event=None):
        """Update project tags."""
        project = self.project_var.get()
        if project:
            tags = [t.strip() for t in self.tags_var.get().split(",") if t.strip()]
            if self.db_manager.update_project_metadata(project, tags=tags):
                self._update_project_stats()

    def _search_messages(self, event=None):
        """Search messages in current project."""
        query = self.search_var.get().strip()
        if not query:
            return
            
        project = self.project_var.get()
        if not project:
            return
            
        results = self.db_manager.search_messages(project, query)
        
        # Clear chat display
        self.chat_display.delete("1.0", tk.END)
        
        # Show search results
        self._add_system_message(f"Search results for: {query}")
        for result in results:
            self._add_user_message(result['prompt'])
            self._add_assistant_message(result['response'])

    def _clear_chat(self):
        """Clear the chat display."""
        if messagebox.askyesno(
            "Confirm Clear",
            "Are you sure you want to clear the chat display?\nThis won't delete any saved messages."
        ):
            self.chat_display.delete("1.0", tk.END)
            self._add_system_message("Chat cleared")

    def run(self):
        """Start the UI."""
        self.root.mainloop()

def main():
    setup_logging()
    
    try:
        # Initialize components
        config = Config()
        db_manager = DatabaseManager()
        llama_api = LLaMAAPI(config, db_manager)  # Pass db_manager to LLaMAAPI
        
        # Create and run the UI
        ui = ChatbotUI(llama_api, db_manager)
        ui.run()
        
    except Exception as e:
        logging.error(f"Error in main execution: {str(e)}")
        logging.error("Error details:", exc_info=True)
        messagebox.showerror(
            "Error",
            f"An error occurred while starting the application:\n{str(e)}"
        )
        sys.exit(1)

if __name__ == "__main__":
    main()