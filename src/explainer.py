import numpy as np

class AlgorithmExplainer:
    """
    Cung cấp các báo cáo phân tích Markdown/LaTeX động giải thích cách tính điểm dự đoán cho Top 1 phim.
    """
    @staticmethod
    def explain_user_based_cf(model, user_idx: int, item_idx: int) -> str:
        mode = model.prediction_mode
        train_matrix = model.train_matrix
        similarity_matrix = model.similarity_matrix
        user_means = model.user_means
        baseline_predictor = model.baseline_predictor
        
        other_users = np.where(train_matrix[:, item_idx] > 0)[0]
        base_rating = user_means[user_idx] if user_means[user_idx] > 0 else 3.0
        
        if len(other_users) == 0:
            return f"Không có láng giềng nào đánh giá phim này. Điểm cơ sở: {base_rating:.2f}"
            
        similarities = similarity_matrix[user_idx, other_users]
        top_indices = np.argsort(similarities)[::-1][:model.k_neighbors]
        top_similarities = similarities[top_indices]
        top_other_users = other_users[top_indices]
        
        sim_sum = np.sum(np.abs(top_similarities))
        if sim_sum == 0:
            return f"Tổng độ tương đồng bằng 0. Điểm cơ sở: {base_rating:.2f}"
            
        ratings = train_matrix[top_other_users, item_idx]
        
        table_md = "| Láng Giềng (User ID) | Tương đồng (Sim) | Đánh giá (R) | "
        pred = 0.0
        formula = ""
        if mode == 'basic':
            table_md += "|\n|---|---|---|---|\n"
            for u, sim, r in zip(top_other_users, top_similarities, ratings):
                table_md += f"| User {u+1} | {sim:.3f} | {r:.1f} | |\n"
            
            raw_pred = np.sum(top_similarities * ratings) / sim_sum
            pred = np.clip(raw_pred, 1.0, 5.0)
            formula = f"$$ \\text{{Pred}} = \\frac{{\\sum (\\text{{Sim}} \\times R)}}{{\\sum |\\text{{Sim}}|}} = {raw_pred:.2f} $$"
            if abs(raw_pred - pred) > 0.01:
                formula += f"\n\n**Điểm sau khi giới hạn (Clip 1-5):** {pred:.2f}"
            
        elif mode == 'means':
            table_md += "Mean Láng Giềng | Độ lệch ($R - Mean$) |\n|---|---|---|---|---|\n"
            means = user_means[top_other_users]
            rating_diffs = ratings - means
            for u, sim, r, m, diff in zip(top_other_users, top_similarities, ratings, means, rating_diffs):
                table_md += f"| User {u+1} | {sim:.3f} | {r:.1f} | {m:.2f} | {diff:+.2f} |\n"
                
            raw_pred = user_means[user_idx] + (np.sum(top_similarities * rating_diffs) / sim_sum)
            pred = np.clip(raw_pred, 1.0, 5.0)
            formula = f"""
Mean của User hiện tại: **{user_means[user_idx]:.2f}**

$$ \\text{{Pred}} = \\text{{Mean}}_{{user}} + \\frac{{\\sum \\text{{Sim}} \\times (R - \\text{{Mean}}_{{neighbor}})}}{{\\sum |\\text{{Sim}}|}} = {raw_pred:.2f} $$
"""
            if abs(raw_pred - pred) > 0.01:
                formula += f"**Điểm sau khi giới hạn (Clip 1-5):** {pred:.2f}\n"
        elif mode == 'biased_baseline':
            table_md += "Bias Láng Giềng ($b_{{vi}}$) | Độ lệch ($R - b_{{vi}}$) |\n|---|---|---|---|---|\n"
            b_ui = baseline_predictor.predict_rating(user_idx, item_idx)
            b_vi = np.array([baseline_predictor.predict_rating(v, item_idx) for v in top_other_users])
            rating_diffs = ratings - b_vi
            for u, sim, r, b, diff in zip(top_other_users, top_similarities, ratings, b_vi, rating_diffs):
                table_md += f"| User {u+1} | {sim:.3f} | {r:.1f} | {b:.2f} | {diff:+.2f} |\n"
                
            raw_pred = b_ui + (np.sum(top_similarities * rating_diffs) / sim_sum)
            pred = np.clip(raw_pred, 1.0, 5.0)
            formula = f"""
Bias dự đoán cơ sở của User hiện tại ($b_{{ui}}$): **{b_ui:.2f}**

$$ \\text{{Pred}} = b_{{ui}} + \\frac{{\\sum \\text{{Sim}} \\times (R - b_{{vi}})}}{{\\sum |\\text{{Sim}}|}} = {raw_pred:.2f} $$
"""
            if abs(raw_pred - pred) > 0.01:
                formula += f"**Điểm sau khi giới hạn (Clip 1-5):** {pred:.2f}\n"
            
        return f"""
**Phân tích: User-Based CF (Chế độ {mode.capitalize()})**

Bảng phân tích chi tiết {len(top_other_users)} láng giềng gần nhất có ảnh hưởng đến kết quả:

{table_md}

**Công thức tính:**
{formula}
"""

    @staticmethod
    def explain_item_based_cf(model, user_idx: int, item_idx: int, movie_titles: dict) -> str:
        mode = model.prediction_mode
        train_matrix = model.train_matrix
        similarity_matrix = model.similarity_matrix
        baseline_predictor = model.baseline_predictor
        
        rated_items = np.where(train_matrix[user_idx, :] > 0)[0]
        if len(rated_items) == 0:
            return "User chưa đánh giá phim nào."
            
        similarities = similarity_matrix[item_idx, rated_items]
        top_k_idx = np.argsort(similarities)[-model.k_neighbors:]
        top_k_idx = top_k_idx[::-1]
        
        top_sims = similarities[top_k_idx]
        top_rated_items = rated_items[top_k_idx]
        
        sim_sum = np.sum(np.abs(top_sims))
        if sim_sum == 0:
            return "Tổng độ tương đồng bằng 0."
            
        ratings = train_matrix[user_idx, top_rated_items]
        
        table_md = "| Phim láng giềng (Item) | Tương đồng (Sim) | Điểm User đã chấm (R) | "
        pred = 0.0
        formula = ""
        if mode == 'basic':
            table_md += "|\n|---|---|---|---|\n"
            for i, sim, r in zip(top_rated_items, top_sims, ratings):
                name = f"[{int(i)+1}] " + movie_titles.get(int(i)+1, f"Phim {i+1}")
                table_md += f"| {name} | {sim:.3f} | {r:.1f} | |\n"
                
            raw_pred = np.sum(top_sims * ratings) / sim_sum
            pred = np.clip(raw_pred, 1.0, 5.0)
            formula = f"$$ \\text{{Pred}} = \\frac{{\\sum (\\text{{Sim}} \\times R)}}{{\\sum |\\text{{Sim}}|}} = {raw_pred:.2f} $$"
            if abs(raw_pred - pred) > 0.01:
                formula += f"\n\n**Điểm sau khi giới hạn (Clip 1-5):** {pred:.2f}"
            
        elif mode == 'biased_baseline':
            table_md += "Bias Phim láng giềng ($b_{{uj}}$) | Độ lệch ($R - b_{{uj}}$) |\n|---|---|---|---|---|\n"
            b_ui = baseline_predictor.predict_rating(user_idx, item_idx)
            b_uj = np.array([baseline_predictor.predict_rating(user_idx, j) for j in top_rated_items])
            rating_diffs = ratings - b_uj
            for i, sim, r, b, diff in zip(top_rated_items, top_sims, ratings, b_uj, rating_diffs):
                name = f"[{int(i)+1}] " + movie_titles.get(int(i)+1, f"Phim {i+1}")
                table_md += f"| {name} | {sim:.3f} | {r:.1f} | {b:.2f} | {diff:+.2f} |\n"
                
            raw_pred = b_ui + (np.sum(top_sims * rating_diffs) / sim_sum)
            pred = np.clip(raw_pred, 1.0, 5.0)
            formula = f"""
Bias dự đoán cơ sở của phim đang xét ($b_{{ui}}$): **{b_ui:.2f}**

$$ \\text{{Pred}} = b_{{ui}} + \\frac{{\\sum \\text{{Sim}} \\times (R - b_{{uj}})}}{{\\sum |\\text{{Sim}}|}} = {raw_pred:.2f} $$
"""
            if abs(raw_pred - pred) > 0.01:
                formula += f"**Điểm sau khi giới hạn (Clip 1-5):** {pred:.2f}\n"
        return f"""
**Phân tích: Item-Based CF (Chế độ {mode.capitalize()})**

Bảng phân tích {len(top_rated_items)} phim láng giềng tương đồng nhất mà user đã xem:

{table_md}

**Công thức tính:**
{formula}
"""

    @staticmethod
    def explain_svd(model, user_idx: int, item_idx: int) -> str:
        mu = model.mu
        b_u = model.b_u[user_idx] if model.b_u is not None else 0.0
        b_i = model.b_i[item_idx] if model.b_i is not None else 0.0
        
        p_u = model.P[user_idx] if model.P is not None else np.zeros(2)
        q_i = model.Q[item_idx] if model.Q is not None else np.zeros(2)
        
        table_md = "| Thành phần | Giá trị |\n|---|---|\n"
        table_md += f"| Global Mean ($\\mu$) | {mu:.4f} |\n"
        table_md += f"| User Bias ($b_u$) | {b_u:.4f} |\n"
        table_md += f"| Item Bias ($b_i$) | {b_i:.4f} |\n"
        
        factors = min(5, len(p_u))
        table_md += "\n**Phân tích Vector đặc trưng ẩn (hiển thị 5 chiều đầu tiên):**\n"
        table_md += "| Latent Factor | $P_u$ (User) | $Q_i$ (Item) | Tích vô hướng ($P_u \\times Q_i$) |\n|---|---|---|---|\n"
        
        dot_product = np.dot(p_u, q_i)
        for i in range(factors):
            table_md += f"| Factor {i+1} | {p_u[i]:.4f} | {q_i[i]:.4f} | {p_u[i]*q_i[i]:.4f} |\n"
            
        raw_pred = mu + b_u + b_i + dot_product
        final_pred = np.clip(raw_pred, 1.0, 5.0)
        
        formula = f"""
**Phân tích: SVD (Matrix Factorization)**

Bảng phân tích các hệ số Bias:

{table_md}

Tích vô hướng đầy đủ ($P_u \\cdot Q_i$): **{dot_product:.4f}**

**Công thức tổng quát:**
$$ \\text{{Pred}} = \\mu + b_u + b_i + (P_u \\cdot Q_i) $$

$$ \\text{{Pred}} = {mu:.2f} + {b_u:.2f} + {b_i:.2f} + {dot_product:.2f} = {raw_pred:.2f} $$
"""
        if abs(raw_pred - final_pred) > 0.01:
            formula += f"\n**Điểm sau khi giới hạn (Clip 1-5):** {final_pred:.2f}\n"
            
        return formula

    @staticmethod
    def _get_sub_matrix_df(train_matrix, target_user, neighbors, target_item, movie_titles, is_item_based=False):
        import pandas as pd
        
        if not is_item_based:
            users_to_include = [target_user] + list(neighbors)
            sub_ratings = train_matrix[users_to_include, :]
            item_counts = np.sum(sub_ratings > 0, axis=0)
            item_counts[target_item] = -1 # Exclude target
            
            top_items = np.argsort(item_counts)[::-1]
            top_items = [i for i in top_items if item_counts[i] > 0]
            
            all_items = [target_item] + top_items
            
            data = []
            for u in users_to_include:
                row = []
                for i in all_items:
                    r = train_matrix[u, i]
                    row.append(r if r > 0 else np.nan)
                data.append(row)
                
            index_labels = [f"User {target_user+1}"] + [f"User {u+1}" for u in neighbors]
            col_labels = [f"[{target_item+1}] " + movie_titles.get(target_item+1, f"Phim {target_item+1}")] + [f"[{i+1}] " + movie_titles.get(i+1, f"Phim {i+1}") for i in top_items]
            
            df = pd.DataFrame(data, index=index_labels, columns=col_labels)
            return df, index_labels, col_labels
        else:
            # Item-based: rows are target_item + neighbor items, cols are target_user + some other users
            items_to_include = [target_item] + list(neighbors)
            sub_ratings = train_matrix[:, items_to_include]
            user_counts = np.sum(sub_ratings > 0, axis=1)
            user_counts[target_user] = -1
            
            top_users = np.argsort(user_counts)[::-1]
            top_users = [u for u in top_users if user_counts[u] > 0]
            
            all_users = [target_user] + top_users
            
            data = []
            for i in items_to_include:
                row = []
                for u in all_users:
                    r = train_matrix[u, i]
                    row.append(r if r > 0 else np.nan)
                data.append(row)
                
            index_labels = [f"[{target_item+1}] " + movie_titles.get(target_item+1, f"Phim {target_item+1}")] + [f"[{i+1}] " + movie_titles.get(i+1, f"Phim {i+1}") for i in neighbors]
            col_labels = [f"User {target_user+1}"] + [f"User {u+1}" for u in top_users]
            
            df = pd.DataFrame(data, index=index_labels, columns=col_labels)
            return df, index_labels, col_labels

    @staticmethod
    def get_user_based_viz_data(model, user_idx: int, item_idx: int, movie_titles: dict) -> dict:
        mode = model.prediction_mode
        train_matrix = model.train_matrix
        similarity_matrix = model.similarity_matrix
        user_means = model.user_means
        baseline_predictor = model.baseline_predictor
        
        other_users = np.where(train_matrix[:, item_idx] > 0)[0]
        base_rating = user_means[user_idx] if user_means[user_idx] > 0 else 3.0
        
        if len(other_users) == 0:
            return {"error": f"Không có láng giềng nào đánh giá phim này. Điểm cơ sở: {base_rating:.2f}"}
            
        similarities = similarity_matrix[user_idx, other_users]
        top_indices = np.argsort(similarities)[::-1][:model.k_neighbors]
        top_similarities = similarities[top_indices]
        top_other_users = other_users[top_indices]
        
        sim_sum = np.sum(np.abs(top_similarities))
        if sim_sum == 0:
            return {"error": f"Tổng độ tương đồng bằng 0. Điểm cơ sở: {base_rating:.2f}"}
            
        ratings = train_matrix[top_other_users, item_idx]
        
        df_matrix, row_labels, col_labels = AlgorithmExplainer._get_sub_matrix_df(train_matrix, user_idx, top_other_users, item_idx, movie_titles)
        
        step1_data = {
            "df_matrix": df_matrix,
            "target_row": row_labels[0],
            "target_col": col_labels[0]
        }
        
        neighbors_data = []
        for u, sim in zip(top_other_users, top_similarities):
            neighbors_data.append({"User ID": f"User {u+1}", "Similarity": sim})
            
        step2_data = {
            "neighbors_data": neighbors_data
        }
        
        step3_data = {"mode": mode, "details": []}
        formula_data = {}
        
        pred = 0.0
        if mode == 'basic':
            raw_pred = np.sum(top_similarities * ratings) / sim_sum
            pred = float(np.clip(raw_pred, 1.0, 5.0))
            for u, sim, r in zip(top_other_users, top_similarities, ratings):
                step3_data["details"].append({"User ID": f"User {u+1}", "Similarity": sim, "Rating": r})
            formula_latex = f"$$ \\text{{Pred}} = \\frac{{\\sum (\\text{{Sim}} \\times R)}}{{\\sum |\\text{{Sim}}|}} = {raw_pred:.2f} $$"
            if abs(raw_pred - pred) > 0.01:
                formula_latex += f"\n\n**Điểm sau khi giới hạn (Clip 1-5):** {pred:.2f}"
            formula_data["formula_latex"] = formula_latex
        elif mode == 'means':
            means = user_means[top_other_users]
            rating_diffs = ratings - means
            for u, sim, r, m, diff in zip(top_other_users, top_similarities, ratings, means, rating_diffs):
                step3_data["details"].append({"User ID": f"User {u+1}", "Similarity": sim, "Rating": r, "Mean": m, "Độ lệch": diff})
            raw_pred = user_means[user_idx] + (np.sum(top_similarities * rating_diffs) / sim_sum)
            pred = float(np.clip(raw_pred, 1.0, 5.0))
            formula_data["user_mean"] = user_means[user_idx]
            formula_latex = f"$$ \\text{{Pred}} = \\text{{Mean}}_{{user}} + \\frac{{\\sum \\text{{Sim}} \\times (R - \\text{{Mean}}_{{neighbor}})}}{{\\sum |\\text{{Sim}}|}} = {raw_pred:.2f} $$"
            if abs(raw_pred - pred) > 0.01:
                formula_latex += f"\n\n**Điểm sau khi giới hạn (Clip 1-5):** {pred:.2f}"
            formula_data["formula_latex"] = formula_latex
        elif mode == 'biased_baseline':
            b_ui = baseline_predictor.predict_rating(user_idx, item_idx)
            b_vi = np.array([baseline_predictor.predict_rating(v, item_idx) for v in top_other_users])
            rating_diffs = ratings - b_vi
            for u, sim, r, b, diff in zip(top_other_users, top_similarities, ratings, b_vi, rating_diffs):
                step3_data["details"].append({"User ID": f"User {u+1}", "Similarity": sim, "Rating": r, "Bias": b, "Độ lệch": diff})
            raw_pred = b_ui + (np.sum(top_similarities * rating_diffs) / sim_sum)
            pred = float(np.clip(raw_pred, 1.0, 5.0))
            formula_data["user_bias"] = b_ui
            formula_latex = f"$$ \\text{{Pred}} = b_{{ui}} + \\frac{{\\sum \\text{{Sim}} \\times (R - b_{{vi}})}}{{\\sum |\\text{{Sim}}|}} = {raw_pred:.2f} $$"
            if abs(raw_pred - pred) > 0.01:
                formula_latex += f"\n\n**Điểm sau khi giới hạn (Clip 1-5):** {pred:.2f}"
            formula_data["formula_latex"] = formula_latex
            
        step4_data = {
            "formula_data": formula_data,
            "pred": pred
        }
            
        return {
            "mode": mode,
            "algo_type": "User-Based CF",
            "step1_data": step1_data,
            "step2_data": step2_data,
            "step3_data": step3_data,
            "step4_data": step4_data
        }

    @staticmethod
    def get_item_based_viz_data(model, user_idx: int, item_idx: int, movie_titles: dict) -> dict:
        mode = model.prediction_mode
        train_matrix = model.train_matrix
        similarity_matrix = model.similarity_matrix
        baseline_predictor = model.baseline_predictor
        
        rated_items = np.where(train_matrix[user_idx, :] > 0)[0]
        if len(rated_items) == 0:
            return {"error": "User chưa đánh giá phim nào."}
            
        similarities = similarity_matrix[item_idx, rated_items]
        top_k_idx = np.argsort(similarities)[-model.k_neighbors:][::-1]
        top_sims = similarities[top_k_idx]
        top_rated_items = rated_items[top_k_idx]
        
        sim_sum = np.sum(np.abs(top_sims))
        if sim_sum == 0:
            return {"error": "Tổng độ tương đồng bằng 0."}
            
        ratings = train_matrix[user_idx, top_rated_items]
        
        df_matrix, row_labels, col_labels = AlgorithmExplainer._get_sub_matrix_df(train_matrix, user_idx, top_rated_items, item_idx, movie_titles, is_item_based=True)
        
        step1_data = {
            "df_matrix": df_matrix,
            "target_row": row_labels[0],
            "target_col": col_labels[0]
        }
        
        neighbors_data = []
        for i, sim in zip(top_rated_items, top_sims):
            name = f"[{int(i)+1}] " + movie_titles.get(int(i)+1, f"Phim {i+1}")
            neighbors_data.append({"Item": name, "Similarity": sim})
            
        step2_data = {
            "neighbors_data": neighbors_data
        }
        
        step3_data = {"mode": mode, "details": []}
        formula_data = {}
        
        pred = 0.0
        if mode == 'basic':
            raw_pred = np.sum(top_sims * ratings) / sim_sum
            pred = float(np.clip(raw_pred, 1.0, 5.0))
            for i, sim, r in zip(top_rated_items, top_sims, ratings):
                name = f"[{int(i)+1}] " + movie_titles.get(int(i)+1, f"Phim {i+1}")
                step3_data["details"].append({"Item": name, "Similarity": sim, "Rating": r})
            formula_latex = f"$$ \\text{{Pred}} = \\frac{{\\sum (\\text{{Sim}} \\times R)}}{{\\sum |\\text{{Sim}}|}} = {raw_pred:.2f} $$"
            if abs(raw_pred - pred) > 0.01:
                formula_latex += f"\n\n**Điểm sau khi giới hạn (Clip 1-5):** {pred:.2f}"
            formula_data["formula_latex"] = formula_latex
        elif mode == 'biased_baseline':
            b_ui = baseline_predictor.predict_rating(user_idx, item_idx)
            b_uj = np.array([baseline_predictor.predict_rating(user_idx, j) for j in top_rated_items])
            rating_diffs = ratings - b_uj
            for i, sim, r, b, diff in zip(top_rated_items, top_sims, ratings, b_uj, rating_diffs):
                name = f"[{int(i)+1}] " + movie_titles.get(int(i)+1, f"Phim {i+1}")
                step3_data["details"].append({"Item": name, "Similarity": sim, "Rating": r, "Bias": b, "Độ lệch": diff})
            raw_pred = b_ui + (np.sum(top_sims * rating_diffs) / sim_sum)
            pred = float(np.clip(raw_pred, 1.0, 5.0))
            formula_data["item_bias"] = b_ui
            formula_latex = f"$$ \\text{{Pred}} = b_{{ui}} + \\frac{{\\sum \\text{{Sim}} \\times (R - b_{{uj}})}}{{\\sum |\\text{{Sim}}|}} = {raw_pred:.2f} $$"
            if abs(raw_pred - pred) > 0.01:
                formula_latex += f"\n\n**Điểm sau khi giới hạn (Clip 1-5):** {pred:.2f}"
            formula_data["formula_latex"] = formula_latex
            
        step4_data = {
            "formula_data": formula_data,
            "pred": pred
        }
            
        return {
            "mode": mode,
            "algo_type": "Item-Based CF",
            "step1_data": step1_data,
            "step2_data": step2_data,
            "step3_data": step3_data,
            "step4_data": step4_data
        }

    @staticmethod
    def get_svd_viz_data(model, user_idx: int, item_idx: int, movie_titles: dict) -> dict:
        mu = model.mu
        b_u = model.b_u[user_idx] if model.b_u is not None else 0.0
        b_i = model.b_i[item_idx] if model.b_i is not None else 0.0
        
        p_u = model.P[user_idx] if model.P is not None else np.zeros(2)
        q_i = model.Q[item_idx] if model.Q is not None else np.zeros(2)
        
        step1_data = {
            "mu": mu,
            "b_u": b_u,
            "b_i": b_i
        }
        
        factors = min(10, len(p_u))
        factors_data = []
        dot_product = np.dot(p_u, q_i)
        
        for i in range(factors):
            factors_data.append({
                "Factor": f"F{i+1}",
                "User Feature (P_u)": p_u[i],
                "Item Feature (Q_i)": q_i[i],
                "Match (P_u * Q_i)": p_u[i] * q_i[i]
            })
            
        step2_data = {
            "factors_data": factors_data
        }
            
        final_pred = np.clip(mu + b_u + b_i + dot_product, 1.0, 5.0)
        
        step3_data = {
            "dot_product": dot_product,
            "pred": final_pred,
            "formula_latex": f"$$ \\text{{Pred}} = \\mu + b_u + b_i + (P_u \\cdot Q_i) = {mu:.2f} + {b_u:.2f} + {b_i:.2f} + {dot_product:.2f} = {final_pred:.2f} $$"
        }
        
        return {
            "mode": "SVD",
            "algo_type": "SVD",
            "step1_data": step1_data,
            "step2_data": step2_data,
            "step3_data": step3_data
        }
