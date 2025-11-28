const auth_user = document.getElementById("authUser");
const auth_pass = document.getElementById("authPass");
const login_btn = document.getElementById("loginBtn");
const create_btn = document.getElementById("createBtn");
const create_section = document.getElementById("createAccount");
const auth_msg = document.getElementById("authMsg");
const who_el = document.getElementById("who");
const logout_btn = document.getElementById("logoutBtn");
const admin_panel = document.getElementById("adminPanel");
const tutorial_btn = document.getElementById("tutorialBtn"); 

function load_users() { // We use local storage to simulate a user database, (we dont use a "real" database)
  try {
    const raw = localStorage.getItem("no_quiz_users");
    return raw ? JSON.parse(raw) : [];
  } catch (e) { return []; }
}

function save_users(users) {
  try {
    localStorage.setItem("no_quiz_users", JSON.stringify(users));
  } catch(e){console.warn("saveUsers failed",e);} 
}

function hash_pass(p) { try { return btoa(p); } catch (e) { return p; } } //Basic base 64 hash to not store plain text, but still obviosuly not for actual production

function find_user(username){
  const users = load_users();
  return users.find(u => u.username === username);
}

function set_current(username){ // session storage handles the "currently logged in" state
  if (username) sessionStorage.setItem("no_quiz_current", username);
  else sessionStorage.removeItem("no_quiz_current");
}

function get_current_user(){
  const u = sessionStorage.getItem("no_quiz_current");
  if (!u) return null;
  return find_user(u) || null;
}

function create_user(username,password,is_admin){ // Logic for making a new user
  username = (username||"").trim();
  if (!username) return {ok:false,msg:"Username required"};
  const users = load_users();
  if (users.find(x=>x.username===username)) return {ok:false,msg:"User exists"};
  const user = { username, pass: hash_pass(password||""), isAdmin: !!is_admin, stats: [] };
  users.push(user);
  save_users(users);
  return {ok:true,user};
}

function login_user(username,password){ // Simple login check
  const u = find_user(username);
  if (!u) return {ok:false,msg:"User not found"};
  if (u.pass !== hash_pass(password||"")) return {ok:false,msg:"Bad password"};
  set_current(u.username);
  return {ok:true,user:u};
}

function logout_user(){ set_current(null); update_auth_ui(); }

function update_auth_ui(){ // toggles visibility of login screen
  const cur = get_current_user();
  const sidebar_content = document.getElementById("sidebarContent");
  if (cur){
    who_el.textContent = cur.username + (cur.isAdmin ? " (admin)" : "");
    logout_btn.classList.remove("hidden");
    tutorial_btn.classList.remove("hidden");
    document.getElementById("auth").classList.add("hidden");
    
    start_mode(); // Logged in, start main interface
    
    if (sidebar_content) sidebar_content.classList.remove("hidden");
    
    if (cur.isAdmin) admin_panel.classList.remove("hidden"); else admin_panel.classList.add("hidden");
  } else {
    who_el.textContent = "";
    logout_btn.classList.add("hidden");
    tutorial_btn.classList.add("hidden");
    document.getElementById("auth").classList.remove("hidden");
    
    if (sidebar_content) sidebar_content.classList.add("hidden");
    
    admin_panel.classList.add("hidden");
  }
}

if (login_btn) login_btn.addEventListener("click", () => {
  const r = login_user(auth_user.value.trim(), auth_pass.value);
  if (r.ok) { auth_msg.textContent = ""; update_auth_ui(); } else { auth_msg.textContent = r.msg; }
});
if (logout_btn) logout_btn.addEventListener("click", () => { logout_user(); });
if (create_btn) create_btn.addEventListener("click", () => {
    document.getElementById("createAccount").classList.remove("hidden");
});
if (document.getElementById("createConfirmBtn")) document.getElementById("createConfirmBtn").addEventListener("click", () => {
    const r = create_user(document.getElementById("createUser").value, document.getElementById("createPass").value, false);
    if(r.ok) { set_current(r.user.username); update_auth_ui(); } else { document.getElementById("createMsg").textContent = r.msg; }
});

update_auth_ui();
 // Main system

const API = "http://localhost:5000";
let app_state = "IDLE"; 
let session = null;
let current_ui_text = {}; 

const ui = {
    box: document.getElementById("sentenceContainer"), // UI references
    btn: document.getElementById("mainBtn"),
    help_btn: document.getElementById("helpBtn"), 
    peek_btn: document.getElementById("peekBtn"), 
    peek_box: document.getElementById("peekBox"), 
    fb: document.getElementById("feedbackArea"),
    loader: document.getElementById("loader"),
    scores: document.getElementById("scoreBoard"),
    title: document.getElementById("uiTitle"),
    profile_title: document.getElementById("uiProfileTitle"),
    lbl_you_wrote: document.getElementById("uiYouWrote"),
    lbl_correct: document.getElementById("uiCorrect"),
    diff_select: document.getElementById("diffSelect"),
    how_graded_btn: document.getElementById("howAmIGradedBtn"),
    grading_info_text: document.getElementById("gradingInfoText")
};

function get_user() {
    return sessionStorage.getItem("no_quiz_current") || "default";
}

function start_mode() {
    document.getElementById("hcai-interface").classList.remove("hidden");
    init();
}

async function init() { // get language and scores
    await changeLanguage(); 
    try {
        const res = await fetch(`${API}/scores`, {
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify({ username: get_user() })
        });
        const scores = await res.json();
        update_scores(scores);
    } catch (e) {
        ui.scores.innerHTML = "Couldnt load data, server running?";
    }
}

function get_lang() { return document.getElementById("langSelect").value; }
function get_text(key) { return current_ui_text[key] || key; }

async function changeLanguage() { // Hits backend to get UI strings for selected language
    const lang = get_lang();
    try {
        const res = await fetch(`${API}/localize`, {
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify({ lang: lang })
        });
        current_ui_text = await res.json();
        if(ui.title) ui.title.innerText = get_text("title");
        if(ui.profile_title) ui.profile_title.innerText = get_text("profile");
        ui.lbl_you_wrote.innerText = get_text("youWrote");
        ui.lbl_correct.innerText = get_text("correct");
        update_button_text();
        ui.help_btn.innerText = get_text("help");
        ui.peek_btn.innerText = get_text("peek") || "Peek";
        
        const manage_btn = document.getElementById("manageDataBtn");
        if(manage_btn) manage_btn.innerText = get_text("manageData") || "Manage My Data";
        
        const confirm_btn = document.getElementById("btnDeleteLocal");
        if(confirm_btn) confirm_btn.innerText = get_text("deleteLocal") || "Delete local data and user";

        if(ui.how_graded_btn) ui.how_graded_btn.innerText = get_text("howGraded");
        if(ui.grading_info_text) ui.grading_info_text.innerText = get_text("gradingExplanation");

        // NEW: Translate Info Modal & Data Modal texts
        if(document.getElementById("uiInfoTitle")) document.getElementById("uiInfoTitle").innerText = get_text("infoTitle");
        if(document.getElementById("uiInfoT5Header")) document.getElementById("uiInfoT5Header").innerText = get_text("infoT5Header");
        if(document.getElementById("uiInfoT5Text")) document.getElementById("uiInfoT5Text").innerHTML = get_text("infoT5Text");
        if(document.getElementById("uiInfoClustersHeader")) document.getElementById("uiInfoClustersHeader").innerText = get_text("infoClustersHeader");
        if(document.getElementById("uiInfoClustersText")) document.getElementById("uiInfoClustersText").innerText = get_text("infoClustersText");
        if(document.getElementById("uiInfoDataHeader")) document.getElementById("uiInfoDataHeader").innerText = get_text("infoDataHeader");
        if(document.getElementById("uiInfoDataText")) document.getElementById("uiInfoDataText").innerText = get_text("infoDataText");
        if(document.getElementById("uiInfoGotItBtn")) document.getElementById("uiInfoGotItBtn").innerText = get_text("infoGotIt");
        
        if(document.getElementById("viewMapBtn")) document.getElementById("viewMapBtn").innerText = get_text("viewMap");
        if(document.getElementById("uiDataModalTitle")) document.getElementById("uiDataModalTitle").innerText = get_text("dataModalTitle");
        if(document.getElementById("uiDataModalMeta")) document.getElementById("uiDataModalMeta").innerText = get_text("dataModalText");
        if(document.getElementById("deleteFileBtn")) document.getElementById("deleteFileBtn").innerText = get_text("deleteFile");

        // NEW: Tutorial Translations
        if(document.getElementById("tutorialBtn")) document.getElementById("tutorialBtn").innerText = get_text("tutorialBtn");
        if(document.getElementById("uiTutTitle")) document.getElementById("uiTutTitle").innerText = get_text("tutorialTitle");
        if(document.getElementById("uiTutIntro")) document.getElementById("uiTutIntro").innerText = get_text("tutIntro");
        if(document.getElementById("uiTutProgress")) document.getElementById("uiTutProgress").innerText = get_text("tutProgress");
        if(document.getElementById("uiTutRec")) document.getElementById("uiTutRec").innerText = get_text("tutRec");
        if(document.getElementById("uiTutGroups")) document.getElementById("uiTutGroups").innerText = get_text("tutGroups");
        if(document.getElementById("uiTutGoodToKnowHeader")) document.getElementById("uiTutGoodToKnowHeader").innerText = get_text("tutGoodToKnowHeader");
        if(document.getElementById("uiTutGoodToKnow1")) document.getElementById("uiTutGoodToKnow1").innerText = get_text("tutGoodToKnow1");
        if(document.getElementById("uiTutGoodToKnow2")) document.getElementById("uiTutGoodToKnow2").innerText = get_text("tutGoodToKnow2");
        if(document.getElementById("uiTutGoodToKnow3")) document.getElementById("uiTutGoodToKnow3").innerText = get_text("tutGoodToKnow3");

    } catch (e) { console.error("Localization failed", e); }
}

function update_button_text() { 
    if (ui.btn.disabled) { ui.btn.innerText = get_text("thinking"); return; }
    if (app_state === "IDLE") { ui.btn.innerText = get_text("start"); ui.help_btn.disabled = true; ui.peek_btn.disabled = true; } 
    else if (app_state === "ACTIVE") { ui.btn.innerText = get_text("submit"); ui.help_btn.disabled = false; ui.peek_btn.disabled = false; } 
    else if (app_state === "FEEDBACK") { ui.btn.innerText = get_text("next"); ui.help_btn.disabled = true; ui.peek_btn.disabled = true; }
}

window.changeLanguage = changeLanguage;
window.handleAction = async function() {
    if (app_state === "IDLE" || app_state === "FEEDBACK") await generate_challenge();
    else if (app_state === "ACTIVE") await submit_answer();
}

window.openDataModal = async function() { // Data management
    document.getElementById("dataModal").classList.remove("hidden");
    const list_el = document.getElementById("fileList");
    list_el.innerHTML = "Loading...";
    
    try {
        const res = await fetch(`${API}/list_user_data`, {
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify({ username: get_user() })
        });
        const files = await res.json();
        
        if (files.length === 0) {
            list_el.innerHTML = "<div class=\"meta\">No files found.</div>";
            return;
        }
        
        let html = "";
        files.forEach(f => {
            html += `
            <div class="file-list-item">
                <div><strong>${f.filename}</strong><br><span class="meta">${f.type}</span></div>
                <button class="secondary" onclick="viewFile('${f.filename}')">View / Edit</button>
            </div>`;
        });
        list_el.innerHTML = html;
    } catch(e) {
        list_el.innerHTML = "Error loading files.";
    }
}

window.closeDataModal = function() {
    document.getElementById("dataModal").classList.add("hidden");
}

window.viewFile = async function(filename) {
    document.getElementById("viewFileName").innerText = filename;
    const area = document.getElementById("fileContentArea");
    area.value = "Loading...";
    document.getElementById("fileViewModal").classList.remove("hidden");
    
    // Configure Delete Button
    const del_btn = document.getElementById("deleteFileBtn");
    del_btn.style.display = "inline-block";
    del_btn.onclick = () => deleteFile(filename);

    try {
        const res = await fetch(`${API}/get_file_content`, {
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify({ filename: filename })
        });
        const data = await res.json();
        area.value = data.content || "Empty";
    } catch(e) { area.value = "Error reading file."; }
}

window.viewClusterMap = async function() {
    document.getElementById("viewFileName").innerText = "Magic labels dictionary";
    const area = document.getElementById("fileContentArea");
    area.value = "Loading sorted map...";
    
    document.getElementById("deleteFileBtn").style.display = "none";
    
    document.getElementById("fileViewModal").classList.remove("hidden");
    
    try {
        const res = await fetch(`${API}/get_cluster_map`);
        const data = await res.json();
        area.value = data.content;
    } catch(e) {
        area.value = "Error loading map.";
    }
}

window.closeFileViewModal = function() {
    document.getElementById("fileViewModal").classList.add("hidden");
}

window.deleteFile = async function(filename) {
    if(!confirm(get_text("confirmDelete"))) return;
    
    const user = get_user();
    try {
        const res = await fetch(`${API}/delete_user_data`, {
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify({ username: user }) 
        });
        if(res.ok) {
            alert("File deleted.");
            closeFileViewModal();
            openDataModal();
            // if active score file was deleted, refresh scores
            init();
        }
    } catch(e) { alert("Error deleting file."); }
}

window.deleteLocalData = async function() {
    if(!confirm("Are you sure? This will delete your account and all associated data.")) return;
    
    const currentUser = get_user();
    
    try {
        await fetch(`${API}/delete_user_data`, {
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify({ username: currentUser }) 
        });
    } catch (e) {
        console.error("Backend deletion failed or file didn't exist", e);
    }

    const users = load_users();
    const updatedUsers = users.filter(u => u.username !== currentUser);
    save_users(updatedUsers);

    sessionStorage.removeItem("no_quiz_current");
    location.reload();
}

window.togglePeek = function() {
    ui.peek_box.classList.toggle("hidden");
}

window.toggleGradingInfo = function() {
    document.getElementById("gradingInfo").classList.toggle("hidden");
}

window.toggleTutorialModal = function() {
    const modal = document.getElementById("tutorialModal");
    if(modal) modal.classList.toggle("hidden");
}

// for keeping underscores with input box
window.updateMask = function() {
    const inputs = document.querySelectorAll(".input-gap");
    const masks = document.querySelectorAll(".bg-mask");
    
    inputs.forEach((input, i) => {
        const mask = masks[i];
        if (!input || !mask) return;

        const val = input.value;
        const target_len = session.target.length; 
        
        let mask_str = "";
        for (let k = 0; k < target_len; k++) {
            if (k < val.length) {
                mask_str += " "; 
            } else {
                mask_str += "_"; 
            }
        }
        mask.innerText = mask_str;
    });
}

window.giveHint = function() { // Reveals one letter at a time as hint
    if (app_state !== "ACTIVE" || !session.target) return;
    
    const input = document.querySelector(".input-gap");
    if (!input) return;
    
    const target = session.target;
    let val_chars = input.value.split("");

    while (val_chars.length < target.length) val_chars.push(" ");

    let candidates = [];
    for (let i = 0; i < target.length; i++) {
        const char = val_chars[i];
        if (char === " " || char.toLowerCase() !== target[i].toLowerCase()) {
            candidates.push(i);
        }
    }

    if (candidates.length === 0) return; 

    const rand_idx = candidates[Math.floor(Math.random() * candidates.length)];
    val_chars[rand_idx] = target[rand_idx];

    const new_val = val_chars.join("");
    document.querySelectorAll(".input-gap").forEach(inp => {
        inp.value = new_val;
    });
    
    updateMask(); 
    input.focus();
}

async function generate_challenge() { // Core challenges loop
    const lang = get_lang();
    
    let payload = { 
        difficulty: ui.diff_select.value, 
        lang: lang, 
        username: get_user() 
    };

    set_loading(true);
    ui.fb.style.display = "none";
    ui.peek_box.classList.add("hidden"); 
    
    try {
        const res = await fetch(`${API}/generate`, {
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify(payload)
        });
        const data = await res.json();
        if (data.error) {
            ui.box.innerHTML = `<span style="color:red">${get_text("backendError")} ${data.error}</span>`;
            return;
        }
        session = { 
            id: data.session_id, 
            target: data.target_word_clean,
            full_trans: data.translated_sentence 
        };
        
        render_challenge(data);
        app_state = "ACTIVE";
        ui.btn.style.backgroundColor = "#3b82f6"; 
        ui.help_btn.disabled = false;
        ui.peek_btn.disabled = false;
        
        ui.peek_box.innerHTML = data.masked_translated_sentence;

    } catch (e) {
        ui.box.textContent = get_text("connectionFail");
    } finally { set_loading(false); }
}

async function submit_answer() {
    const input = document.querySelector(".input-gap");
    let val = input ? input.value.trim() : ""; 

    if (!val) return; 

    set_loading(true);
    try {
        const res = await fetch(`${API}/grade`, {
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify({ 
                session_id: session.id, 
                answer: val, 
                username: get_user() 
            })
        });
        const data = await res.json();
        render_feedback(data);
        update_scores(data.scores_snapshot); 
        app_state = "FEEDBACK";
        ui.btn.style.backgroundColor = "#111827"; 
        ui.help_btn.disabled = true;
        ui.peek_btn.disabled = true;
    } catch (e) { console.error(e); } 
    finally { set_loading(false); }
}

function render_challenge(data) { // Injects sentence to input box
    const parts = data.text.split("[___]");
    let html = "";
    const init_mask = Array(session.target.length).fill("_").join("");

    parts.forEach((part, index) => {
        html += `<span>${part}</span>`;
        if (index < parts.length - 1) {
            html += `
            <span class="hint-wrapper">
                <span class="meta-info">
                    <span class="label-tag">${data.label_type}</span>
                    <span class="trans-text">${data.translated_word}</span>
                </span>
                <span class="input-wrapper">
                    <div class="bg-mask">${init_mask}</div>
                    <input type="text" class="input-gap" autocomplete="off" oninput="updateMask()" onkeyup="syncInputs(this)">
                </span>
            </span>`;
        }
    });

    ui.box.innerHTML = html;
    setTimeout(() => { const inp = document.querySelector(".input-gap"); if(inp) inp.focus(); }, 100);
}

window.syncInputs = function(src) {
    const val = src.value;
    document.querySelectorAll(".input-gap").forEach(inp => {
        if (inp !== src) inp.value = val;
    });
    updateMask(); 
}

function render_feedback(data) {
    ui.fb.style.display = "block";
    const is_correct = data.status === "correct";
    ui.fb.className = `feedback-box ${is_correct ? "feedback-correct" : "feedback-wrong"}`;
    
    document.getElementById("fbTitle").textContent = is_correct ? get_text("correctExcl") : get_text("incorrectExcl");
    document.getElementById("fbUser").textContent = data.user_word;
    document.getElementById("fbCorrect").textContent = data.correct_word;
    document.getElementById("fbFullTrans").innerHTML = data.translated_sentence;
    document.getElementById("fbExplain").textContent = `Distance: ${data.distance}. Proficiency for [${data.label}] changed by ${data.score_change}.`;
}

function update_scores(scores) {
    if (!scores || Object.keys(scores).length === 0) {
        ui.scores.innerHTML = `<span style="color:#666; font-size:0.9em">${get_text("noData")}</span>`;
        return;
    }
    ui.scores.innerHTML = "";
    Object.entries(scores).sort((a,b) => b[1] - a[1]).forEach(([label, score]) => {
        const el = document.createElement("span");
        let cls = "tag";
        if (score >= 3.5) cls += " mastered";
        else if (score >= 1.5) cls += " good";
        else if (score >= -1.5) cls += " neutral";
        else if (score >= -3.5) cls += " struggle";
        else cls += " critical";
        el.className = cls;
        el.innerText = `${label}: ${score}`;
        ui.scores.appendChild(el);
    });
}

function set_loading(is_loading) {
    ui.loader.style.display = is_loading ? "inline" : "none";
    ui.btn.disabled = is_loading;
    ui.help_btn.disabled = is_loading || app_state !== "ACTIVE";
    ui.peek_btn.disabled = is_loading || app_state !== "ACTIVE";
    update_button_text(); 
}

window.toggleInfoModal = function() { // info button
    const modal = document.getElementById("infoModal");
    if (modal) modal.classList.toggle("hidden");
}