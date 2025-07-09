import os
import os.path as osp
import sqlite3
import hashlib
import threading
from typing import List
import faiss
import numpy as np
import torch
import clip
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class ImageSearchApp:
    def __init__(self, root):
        self.root = root
        self.root.title("图像搜索工具")
        self.root.geometry("1200x800")
        self.root.configure(bg="#f0f0f0")

        # 初始化模型
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = None, None
        self.thread = None
        self.search_running = False
        self.status_var = tk.StringVar(value="就绪")

        self.create_widgets()
        self.load_clip_model()

    def create_widgets(self):
        # 创建主框架
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # 创建左右分割框架
        paned_window = ttk.PanedWindow(main_frame, orient=tk.HORIZONTAL)
        paned_window.pack(fill=tk.BOTH, expand=True)

        # 左侧控制面板
        left_frame = ttk.Frame(paned_window, width=400)
        paned_window.add(left_frame, weight=1)

        # 右侧结果面板
        right_frame = ttk.Frame(paned_window)
        paned_window.add(right_frame, weight=3)

        # 文件夹选择部分
        folder_frame = ttk.LabelFrame(left_frame, text="文件夹设置")
        folder_frame.pack(fill=tk.X, padx=5, pady=5)

        self.folder_var = tk.StringVar()
        folder_entry = ttk.Entry(folder_frame, textvariable=self.folder_var)
        folder_entry.pack(side=tk.LEFT, padx=5, pady=5, fill=tk.X, expand=True)

        browse_btn = ttk.Button(
            folder_frame, text="浏览", command=self.browse_folder, width=8
        )
        browse_btn.pack(side=tk.RIGHT, padx=5, pady=5)

        # 搜索类型选择
        search_frame = ttk.LabelFrame(left_frame, text="搜索类型")
        search_frame.pack(fill=tk.X, padx=5, pady=5)

        self.search_mode = tk.StringVar(value="text")
        text_rb = ttk.Radiobutton(
            search_frame,
            text="文本搜索",
            variable=self.search_mode,
            value="text",
            command=self.toggle_search_mode,
        )
        text_rb.pack(side=tk.TOP, anchor=tk.W, padx=5, pady=2)

        image_rb = ttk.Radiobutton(
            search_frame,
            text="图像搜索",
            variable=self.search_mode,
            value="image",
            command=self.toggle_search_mode,
        )
        image_rb.pack(side=tk.TOP, anchor=tk.W, padx=5, pady=2)

        # 文本搜索输入
        self.text_frame = ttk.LabelFrame(left_frame, text="文本搜索")
        self.text_frame.pack(fill=tk.X, padx=5, pady=5)

        self.text_var = tk.StringVar()
        text_entry = ttk.Entry(self.text_frame, textvariable=self.text_var)
        text_entry.pack(padx=5, pady=5, fill=tk.X)

        # 图像搜索输入
        self.image_frame = ttk.LabelFrame(left_frame, text="图像搜索")
        self.image_frame.pack(fill=tk.X, padx=5, pady=5)

        image_btn = ttk.Button(
            self.image_frame, text="选择参考图片", command=self.select_image
        )
        image_btn.pack(padx=5, pady=5)

        self.img_preview_label = ttk.Label(self.image_frame)
        self.img_preview_label.pack(padx=5, pady=5)
        self.selected_image_path = None
        self.selected_photo = None

        # 选项部分
        option_frame = ttk.LabelFrame(left_frame, text="搜索选项")
        option_frame.pack(fill=tk.X, padx=5, pady=5)

        self.rebuild_var = tk.BooleanVar()
        rebuild_cb = ttk.Checkbutton(
            option_frame, text="重建索引", variable=self.rebuild_var
        )
        rebuild_cb.pack(side=tk.TOP, anchor=tk.W, padx=5, pady=2)

        result_frame = ttk.Frame(option_frame)
        result_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Label(result_frame, text="结果数量:").pack(side=tk.LEFT)
        self.result_num = tk.StringVar(value="5")
        result_entry = ttk.Entry(result_frame, textvariable=self.result_num, width=5)
        result_entry.pack(side=tk.RIGHT, padx=5)

        # 按钮部分
        button_frame = ttk.Frame(left_frame)
        button_frame.pack(fill=tk.X, padx=5, pady=10)

        self.search_btn = ttk.Button(
            button_frame, text="搜索", command=self.start_search
        )
        self.search_btn.pack(side=tk.LEFT, padx=5, pady=5, fill=tk.X, expand=True)

        clear_btn = ttk.Button(
            button_frame, text="清空结果", command=self.clear_results
        )
        clear_btn.pack(side=tk.RIGHT, padx=5, pady=5, fill=tk.X, expand=True)

        # 结果部分
        result_container = ttk.Frame(right_frame)
        result_container.pack(fill=tk.BOTH, expand=True)

        self.result_frame = ttk.LabelFrame(result_container, text="搜索结果")
        self.result_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # 创建带有滚动条的画布
        self.canvas = tk.Canvas(self.result_frame)
        self.scrollbar = ttk.Scrollbar(
            self.result_frame, orient=tk.VERTICAL, command=self.canvas.yview
        )

        # 创建可滚动的框架
        self.scrollable_frame = ttk.Frame(self.canvas)
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")),
        )

        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        # 布局
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # 状态栏 - 固定在底部
        status_bar = ttk.Frame(self.root, relief=tk.SUNKEN, height=24)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X, padx=0, pady=0)
        status_bar.pack_propagate(False)  # 固定高度

        # 状态标签
        self.status_label = ttk.Label(
            status_bar, textvariable=self.status_var, anchor=tk.W
        )
        self.status_label.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)

        # 初始状态
        self.toggle_search_mode()

    def browse_folder(self):
        folder_selected = filedialog.askdirectory()
        if folder_selected:
            self.folder_var.set(folder_selected)
            self.status_var.set(f"已选择文件夹: {osp.basename(folder_selected)}")

    def toggle_search_mode(self):
        if self.search_mode.get() == "text":
            self.text_frame.pack(fill=tk.X, padx=5, pady=5)
            self.image_frame.pack_forget()
        else:
            self.image_frame.pack(fill=tk.X, padx=5, pady=5)
            self.text_frame.pack_forget()

    def select_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.jpg;*.jpeg;*.png;*.gif;*.webp")]
        )
        if file_path:
            self.selected_image_path = file_path
            try:
                # 显示预览图
                img = Image.open(file_path)
                img.thumbnail((150, 150))

                # 先清理之前的图像引用
                if hasattr(self, "selected_photo"):
                    self.selected_photo = None

                self.selected_photo = ImageTk.PhotoImage(img)
                self.img_preview_label.configure(image=self.selected_photo)
                self.img_preview_label.image = self.selected_photo
                self.status_var.set(f"已选择图片: {osp.basename(file_path)}")
            except Exception as e:
                self.status_var.set(f"无法加载图片: {str(e)}")
                messagebox.showerror("错误", f"无法加载图片: {str(e)}")

    def clear_results(self):
        for widget in self.scrollable_frame.winfo_children():
            widget.destroy()
        self.status_var.set("结果已清空")

    def load_clip_model(self):
        # 在后台线程中加载CLIP模型
        self.search_btn.config(state=tk.DISABLED, text="加载模型中...")
        threading.Thread(target=self._load_model_thread, daemon=True).start()

    def _load_model_thread(self):
        try:
            self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)
            self.root.after(
                0, lambda: self.search_btn.config(state=tk.NORMAL, text="搜索")
            )
            self.root.after(0, lambda: self.status_var.set("模型加载成功"))
        except Exception as e:
            self.root.after(
                0, lambda: messagebox.showerror("错误", f"加载模型失败: {str(e)}")
            )
            self.root.after(
                0, lambda: self.search_btn.config(state=tk.NORMAL, text="搜索")
            )

    def start_search(self):
        if self.search_running:
            self.status_var.set("当前已有搜索正在进行")
            return

        folder = self.folder_var.get()
        if not folder or not os.path.isdir(folder):
            messagebox.showerror("错误", "请选择有效的文件夹")
            return

        if not self.model or not self.preprocess:
            messagebox.showerror("错误", "模型尚未加载完成")
            return

        if self.search_mode.get() == "image" and not self.selected_image_path:
            messagebox.showerror("错误", "请选择参考图片")
            return

        if self.search_mode.get() == "text" and not self.text_var.get().strip():
            messagebox.showerror("错误", "请输入搜索文本")
            return

        try:
            num = int(self.result_num.get())
            if num <= 0:
                raise ValueError("结果数量必须是正整数")
        except ValueError as e:
            messagebox.showerror("错误", str(e))
            return

        self.search_running = True
        self.search_btn.config(state=tk.DISABLED, text="搜索中...")
        self.status_var.set("搜索中...")
        self.clear_results()

        # 启动搜索线程
        self.thread = threading.Thread(
            target=self.search, args=(folder, num), daemon=True
        )
        self.thread.start()
        self.root.after(100, self.check_thread)

    def check_thread(self):
        if self.thread.is_alive():
            self.root.after(100, self.check_thread)
        else:
            self.search_running = False
            self.search_btn.config(state=tk.NORMAL, text="搜索")

    def search(self, img_root, num_results):
        try:
            # 检查是否需要重建索引
            rebuild = self.rebuild_var.get()
            if rebuild:
                self.root.after(0, lambda: self.status_var.set("正在重建索引..."))
                self.rebuild_index(img_root)

            # 检查索引是否存在
            if not self.check_index_exists(img_root) or rebuild:
                self.root.after(0, lambda: self.status_var.set("正在构建索引..."))
                self.build_index(img_root)

            # 执行搜索
            if self.search_mode.get() == "text":
                text = self.text_var.get().strip()
                self.root.after(0, lambda: self.status_var.set(f"正在搜索: {text}"))
                results, scores = self.search_vec_by_text(text, img_root, num_results)
            else:
                img_name = osp.basename(self.selected_image_path)
                self.root.after(
                    0, lambda: self.status_var.set(f"正在搜索匹配图片: {img_name}")
                )
                results, scores = self.search_vec_by_image(
                    self.selected_image_path, img_root, num_results
                )

            self.root.after(0, lambda: self.display_results(results, scores))

        except Exception as e:
            error_msg = f"搜索失败: {str(e)}"
            self.root.after(0, lambda: messagebox.showerror("错误", error_msg))
            self.root.after(0, lambda: self.status_var.set("搜索失败"))

    def check_index_exists(self, img_root):
        idx_path = osp.join(img_root, "faissidx.idx")
        db_path = osp.join(img_root, "imgsnames.db")
        return osp.exists(idx_path) and osp.exists(db_path)

    def rebuild_index(self, img_root):
        """删除旧索引以便重建"""
        try:
            idx_path = osp.join(img_root, "faissidx.idx")
            db_path = osp.join(img_root, "imgsnames.db")

            if osp.exists(idx_path):
                os.remove(idx_path)
            if osp.exists(db_path):
                os.remove(db_path)
            return True
        except Exception as e:
            self.root.after(
                0, lambda: messagebox.showwarning("警告", f"无法删除旧索引: {str(e)}")
            )
            return False

    def build_index(self, img_root):
        """构建新的索引"""
        try:
            self.clipv(img_root)
            return True
        except Exception as e:
            self.root.after(
                0, lambda: messagebox.showerror("错误", f"构建索引失败: {str(e)}")
            )
            return False

    def generate_id(self, string: str):
        sha256 = hashlib.sha256()
        sha256.update(string.encode("utf-8"))
        img_id = int.from_bytes(sha256.digest()[:6], byteorder="big")
        return img_id

    def find_files_with_ext(
        self,
        folder_root: str,
        extensions: List[str] = [
            ".PNG",
            ".JPG",
            ".JPEG",
            ".png",
            ".jpg",
            ".jpeg",
            ".gif",
            ".webp",
        ],
    ):
        file_list = []
        for root, dirs, files in os.walk(folder_root):
            for file in files:
                if any(file.lower().endswith(ext) for ext in extensions):
                    file_list.append(os.path.join(root, file))
        return file_list

    def clipv(self, root: str):
        dimension = 512
        idx_flat = faiss.IndexFlatL2(dimension)
        index = faiss.IndexIDMap(idx_flat)

        faiss_idx_file = osp.join(root, "faissidx.idx")
        db = osp.join(root, "imgsnames.db")

        # 创建文件夹如果不存在
        os.makedirs(root, exist_ok=True)

        img_paths = self.find_files_with_ext(root)
        self.root.after(
            0, lambda: self.status_var.set(f"正在处理 {len(img_paths)} 张图片...")
        )

        if len(img_paths) == 0:
            self.root.after(
                0,
                lambda: messagebox.showwarning("警告", f"在所选文件夹中未找到任何图片"),
            )
            return

        # 如果有现有索引，加载它
        if osp.exists(faiss_idx_file):
            try:
                index = faiss.read_index(faiss_idx_file)
            except:
                pass  # 如果读取失败，则创建一个新的

        conn = sqlite3.connect(db)
        cursor = conn.cursor()
        cursor.execute(
            """CREATE TABLE IF NOT EXISTS images (id INTEGER PRIMARY KEY, name TEXT)"""
        )
        conn.commit()

        # 获取现有ID
        cursor.execute("SELECT id FROM images")
        exist_ids = set(row[0] for row in cursor.fetchall())

        collector = []
        for img_path in img_paths:
            image_id = self.generate_id(img_path)
            if image_id not in exist_ids:
                collector.append(img_path)

        if len(collector) == 0:
            self.root.after(0, lambda: self.status_var.set("索引已是最新，无需更新"))
            conn.close()
            return

        processed, skipped = 0, 0
        for i, img_path in enumerate(collector):
            try:
                img_id = self.generate_id(img_path)
                img = Image.open(img_path)

                # 跳过损坏的图片
                if img is None:
                    skipped += 1
                    continue

                img = self.preprocess(img).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    img_feat = self.model.encode_image(img)
                img_feat_np = img_feat.cpu().numpy()
                norms = np.linalg.norm(img_feat_np, axis=1, keepdims=True)
                img_feat_np /= norms
                index.add_with_ids(img_feat_np, np.array([img_id]))
                cursor.execute(
                    "INSERT INTO images (id, name) VALUES (?, ?)", (img_id, img_path)
                )
                processed += 1

                # 每处理100张图片保存一次进度
                if processed % 100 == 0:
                    conn.commit()
                    faiss.write_index(index, faiss_idx_file)
                    self.root.after(
                        0, lambda: self.status_var.set(f"已处理 {processed} 张图片...")
                    )

            except Exception as e:
                skipped += 1
                # 记录错误日志但不中断
                if processed % 50 == 0:
                    self.root.after(
                        0,
                        lambda: self.status_var.set(
                            f"跳过错误图片: {osp.basename(img_path)}"
                        ),
                    )

        conn.commit()
        faiss.write_index(index, faiss_idx_file)
        conn.close()

        status_msg = f"索引构建完成: 成功处理 {processed} 张图片"
        if skipped > 0:
            status_msg += f", 跳过 {skipped} 张图片"

        self.root.after(0, lambda: self.status_var.set(status_msg))
        return True

    def search_vec_by_text(self, text: str, root: str, num_results: int = 5):
        try:
            text_tokenized = clip.tokenize([text]).to(self.device)
            with torch.no_grad():
                text_feat = self.model.encode_text(text_tokenized)
            text_feat_np = text_feat.cpu().numpy()

            faiss_index_file = osp.join(root, "faissidx.idx")
            if not osp.exists(faiss_index_file):
                self.root.after(
                    0, lambda: self.status_var.set("索引文件不存在，请先构建索引")
                )
                return [], []

            index = faiss.read_index(faiss_index_file)
            norm = np.linalg.norm(text_feat_np, axis=1, keepdims=True)
            text_feat_np /= norm

            db = osp.join(root, "imgsnames.db")
            if not osp.exists(db):
                self.root.after(0, lambda: self.status_var.set("数据库文件不存在"))
                return [], []

            D, I = index.search(text_feat_np, num_results)
            img_paths = []
            similarities = []
            conn = sqlite3.connect(db)
            cursor = conn.cursor()
            for i, idx in enumerate(I[0]):
                cursor.execute("SELECT name FROM images WHERE id = ?", (int(idx),))
                res = cursor.fetchone()
                if res:
                    path = res[0]
                    if osp.exists(path):
                        img_paths.append(path)
                        dist = D[0][i]
                        # 直接使用距离值作为分数（不转换）
                        similarities.append(dist)  # 这里保留原始距离值
            conn.close()
            return img_paths, similarities
        except Exception as e:
            error_msg = f"文本搜索失败: {str(e)}"
            self.root.after(0, lambda: self.status_var.set(error_msg))
            return [], []

    def search_vec_by_image(self, ref_image: str, root: str, num_results: int = 5):
        try:
            img = Image.open(ref_image)
            img = self.preprocess(img).unsqueeze(0).to(self.device)
            with torch.no_grad():
                img_feat = self.model.encode_image(img)
            img_feat_np = img_feat.cpu().numpy()
            norms = np.linalg.norm(img_feat_np, axis=1, keepdims=True)
            img_feat_np /= norms

            faiss_index_file = osp.join(root, "faissidx.idx")
            if not osp.exists(faiss_index_file):
                self.root.after(
                    0, lambda: self.status_var.set("索引文件不存在，请先构建索引")
                )
                return [], []

            index = faiss.read_index(faiss_index_file)
            norm = np.linalg.norm(img_feat_np, axis=1, keepdims=True)
            img_feat_np /= norm

            db = osp.join(root, "imgsnames.db")
            if not osp.exists(db):
                self.root.after(0, lambda: self.status_var.set("数据库文件不存在"))
                return [], []

            D, I = index.search(img_feat_np, num_results)
            img_paths = []
            scores = []
            conn = sqlite3.connect(db)
            cursor = conn.cursor()
            for i, idx in enumerate(I[0]):
                cursor.execute("SELECT name FROM images WHERE id = ?", (int(idx),))
                res = cursor.fetchone()
                if res:
                    path = res[0]
                    if osp.exists(path):  # 检查图片是否仍然存在
                        img_paths.append(path)
                        scores.append(D[0][i])  # 保存距离分数
            conn.close()
            return img_paths, scores
        except Exception as e:
            error_msg = f"图像搜索失败: {str(e)}"
            self.root.after(0, lambda: self.status_var.set(error_msg))
            return [], []

    def display_results(self, img_paths, scores):
        self.clear_results()  # 先清空现有结果

        if not img_paths:
            self.status_var.set("未找到匹配的图片")
            # 创建一个空结果提示
            empty_frame = ttk.Frame(self.scrollable_frame)
            empty_frame.pack(fill=tk.X, padx=10, pady=50)
            ttk.Label(
                empty_frame,
                text="未找到匹配的图片",
                font=("Arial", 14),
                foreground="gray",
            ).pack()
            return

        self.status_var.set(f"找到 {len(img_paths)} 个结果")

        # 创建结果网格布局
        for i, img_path in enumerate(img_paths):
            # 每个结果项使用固定高度的框架
            result_frame = ttk.Frame(self.scrollable_frame, relief=tk.GROOVE, padding=5)
            result_frame.pack(fill=tk.X, padx=5, pady=5, ipady=5)

            try:
                # 加载并显示图片
                img = Image.open(img_path)
                img.thumbnail((150, 150))
                photo = ImageTk.PhotoImage(img)

                # 图片容器
                img_container = ttk.Frame(result_frame)
                img_container.pack(side=tk.LEFT, padx=5, pady=5)

                img_label = ttk.Label(img_container, image=photo)
                img_label.image = photo  # 保持引用
                img_label.pack()

                # 信息容器
                info_frame = ttk.Frame(result_frame)
                info_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)

                # 显示文件名
                filename = osp.basename(img_path)
                filename_label = ttk.Label(
                    info_frame, text=f"文件名: {filename}", wraplength=400
                )
                filename_label.pack(anchor=tk.W)

                # 显示路径(缩写)
                folder_path = osp.dirname(img_path)
                if len(folder_path) > 50:
                    folder_path = "..." + folder_path[-47:]
                path_label = ttk.Label(
                    info_frame, text=f"位置: {folder_path}", wraplength=400
                )
                path_label.pack(anchor=tk.W)

                # 显示相似度分数
                if i < len(scores):
                    score = scores[i]
                    if self.search_mode.get() == "text":
                        # 文本搜索 - 传入的是距离值
                        # 使用对数转换来显示相似度百分比
                        similarity = 100 * np.exp(-score / 2)
                    else:
                        # 图像搜索 - 直接使用距离值
                        # 使用不同的转换公式
                        similarity = 100 * np.exp(-score)

                    # 确保相似度在0-100范围内
                    similarity = max(0, min(100, similarity))
                    score_label = ttk.Label(
                        info_frame,
                        text=f"匹配度: {similarity:.2f}%",
                        font=("Arial", 10, "bold"),
                    )
                    score_label.pack(anchor=tk.W, pady=(5, 0))

                # 添加工具提示显示完整路径
                path_label.bind(
                    "<Enter>", lambda e, path=img_path: self.show_tooltip(e, path)
                )
                path_label.bind("<Leave>", self.hide_tooltip)

                # 按钮容器
                button_frame = ttk.Frame(result_frame)
                button_frame.pack(side=tk.RIGHT, padx=5, pady=5)

                # 添加打开按钮
                open_btn = ttk.Button(
                    button_frame,
                    text="打开图片",
                    command=lambda path=img_path: self.open_image(path),
                    width=10,
                )
                open_btn.pack(pady=2)

                # 添加复制路径按钮
                copy_btn = ttk.Button(
                    button_frame,
                    text="复制路径",
                    command=lambda path=img_path: self.copy_to_clipboard(path),
                    width=10,
                )
                copy_btn.pack(pady=2)

            except Exception as e:
                self.status_var.set(f"无法加载图片: {osp.basename(img_path)}")
                error_label = ttk.Label(
                    result_frame, text=f"无法加载图片\n{osp.basename(img_path)}"
                )
                error_label.pack()
                continue

    def show_tooltip(self, event, path):
        """显示完整路径的工具提示"""
        x, y, _, _ = event.widget.bbox("insert")
        x += event.widget.winfo_rootx() + 25
        y += event.widget.winfo_rooty() + 25

        # 创建工具提示窗口
        self.tooltip = tk.Toplevel(event.widget)
        self.tooltip.wm_overrideredirect(True)
        self.tooltip.wm_geometry(f"+{x}+{y}")

        label = ttk.Label(
            self.tooltip,
            text=path,
            background="#ffffe0",
            relief="solid",
            borderwidth=1,
            padding=5,
            wraplength=400,
        )
        label.pack()

    def hide_tooltip(self, event):
        """隐藏工具提示"""
        if hasattr(self, "tooltip") and self.tooltip:
            self.tooltip.destroy()
            self.tooltip = None

    def copy_to_clipboard(self, text):
        """复制文本到剪贴板"""
        self.root.clipboard_clear()
        self.root.clipboard_append(text)
        self.status_var.set(f"已复制到剪贴板: {osp.basename(text)}")

    def open_image(self, path):
        try:
            self.status_var.set(f"打开图片: {osp.basename(path)}")
            if os.name == "nt":  # Windows
                os.startfile(path)
            elif os.name == "posix":  # Linux/Mac
                os.system(f'xdg-open "{path}"')
            else:
                messagebox.showinfo("打开图片", f"路径: {path}")
        except Exception as e:
            self.status_var.set(f"无法打开图片: {osp.basename(path)}")
            messagebox.showerror("错误", f"无法打开图片: {str(e)}")

    def on_closing(self):
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = ImageSearchApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()
