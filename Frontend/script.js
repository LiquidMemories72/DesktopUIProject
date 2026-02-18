const API = "http://127.0.0.1:8000";

let actionsData = {};
let gesturesData = {};


// 🚀 INITIAL LOAD
window.onload = async () => {
    await loadActions();
    await loadGestures();
    loadStatus();
    setInterval(loadStatus, 2000);
};



// =============================
// LOAD ACTIONS FOR DROPDOWN
// =============================
async function loadActions() {
    actionsData = await fetch(API + "/actions").then(res => res.json());
}



// =============================
// LOAD GESTURES → BUILD CARDS
// =============================
async function loadGestures() {

    gesturesData = await fetch(API + "/gestures").then(res => res.json());

    const grid = document.getElementById("gestureGrid");
    grid.innerHTML = "";

    for (const gesture in gesturesData) {
        createCard(gesture, gesturesData[gesture]);
    }
}



// =============================
// CREATE GESTURE CARD
// =============================
function createCard(name, currentAction) {

    const card = document.createElement("div");
    card.className = "card";
    card.id = "card-" + name;

    const title = document.createElement("h3");
    title.innerText = name;

    const dropdown = document.createElement("select");
    
    for (const action in actionsData) {

        const option = document.createElement("option");
        option.value = action;
        option.text = actionsData[action].label;

        if (action === currentAction) option.selected = true;

        dropdown.appendChild(option);
    }

    // ACTIONS CONTAINER
    const actionsDiv = document.createElement("div");
    actionsDiv.className = "card-actions";

    // 📸 CAPTURE BUTTON
    const captureBtn = document.createElement("button");
    captureBtn.innerText = "📸";
    captureBtn.title = "Capture Gesture Images";
    captureBtn.onclick = async () => {

        setStatus("Capturing " + name + "...");
        await fetch(API + "/capture/" + name, { method: "POST" });
        setStatus("Capture complete");
    };


    // 💾 SAVE MAPPING
    const saveBtn = document.createElement("button");
    saveBtn.innerText = "💾";
    saveBtn.title = "Save Mapping";
    saveBtn.className = "save-btn";
    saveBtn.onclick = async () => {

        await fetch(API + "/map-gesture", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                gesture: name,
                action: dropdown.value
            })
        });

        flashCard(name);
    };


    // 🗑 DELETE
    const deleteBtn = document.createElement("button");
    deleteBtn.innerText = "🗑";
    deleteBtn.title = "Delete Gesture";
    deleteBtn.className = "delete-btn";
    deleteBtn.onclick = async () => {

        if(!confirm("Are you sure you want to delete gesture '" + name + "'?")) return;

        await fetch(API + "/delete-gesture/" + name, { method: "POST" });
        loadGestures();
    };

    actionsDiv.appendChild(captureBtn);
    actionsDiv.appendChild(saveBtn);
    actionsDiv.appendChild(deleteBtn);

    card.appendChild(title);
    card.appendChild(dropdown);
    card.appendChild(actionsDiv);

    document.getElementById("gestureGrid").appendChild(card);
}



// =============================
// ADD NEW GESTURE
// =============================
async function addGesture() {

    const name = document.getElementById("newGesture").value;

    if (!name) return;

    await fetch(API + "/add-gesture", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ gesture: name })
    });

    document.getElementById("newGesture").value = "";

    loadGestures();
}



// =============================
// TRAIN MODEL
// =============================
async function trainModel() {

    setStatus("Training model...");

    await fetch(API + "/train", { method: "POST" });

    setStatus("Model ready ✅");
}



// =============================
// SYSTEM STATUS
// =============================
async function loadStatus() {

    const status = await fetch(API + "/status").then(res => res.json());

    document.getElementById("systemStatus").innerText =
        "System: " + status.status +
        (status.last_action ? " | Last: " + status.last_action : "");

    if (status.last_action) {

        const gesture = status.last_action.split(" → ")[0];
        flashCard(gesture);
    }
}



function setStatus(msg) {
    document.getElementById("systemStatus").innerText = msg;
}



// =============================
// FLASH ANIMATION
// =============================
function flashCard(name) {

    const card = document.getElementById("card-" + name);

    if (!card) return;

    card.classList.add("flash");

    setTimeout(() => {
        card.classList.remove("flash");
    }, 600);
}
async function startAI() {
    await fetch(API + "/start-detection", { method: "POST" });
}

async function stopAI() {
    await fetch(API + "/stop-detection", { method: "POST" });
}

