import { el, fmtNum } from "./app.js";

const CLOSED_BY_DEFAULT = new Set(["system", "tool"]);

// row: one trajectory-sidecar entry {status, reward, messages: [...]}
// (OpenAI message shape: role/content plus optional reasoning_content,
// tool_calls, name)
export function renderConversation(row) {
  const chips = [el("span", { class: "chip" }, [row.status])];
  if (row.reward !== null && row.reward !== undefined) {
    chips.push(el("span", { class: "chip" }, [`reward ${fmtNum(row.reward)}`]));
  }
  return el("div", { class: "panel" }, [
    el("h3", {}, ["Conversation"]),
    el("div", { class: "controls" }, chips),
    ...row.messages.map(messageCard),
  ]);
}

// display-level split of raw assistant text (engine multi_turn path records
// the model's verbatim emission): <think> blocks, <tool_call> markup, and
// trailing special tokens; unknown markup stays visible as-is
function splitRawAssistant(text) {
  const thinking = [];
  let rest = text.replace(/<think>([\s\S]*?)<\/think>/g, (_, body) => {
    thinking.push(body.trim());
    return "";
  });
  const calls = [];
  rest = rest.replace(/<tool_call>\s*([\s\S]*?)\s*<\/tool_call>/g, (_, body) => {
    calls.push(body);
    return "";
  });
  rest = rest.replace(/(?:<\|[^|>]+\|>\s*)+$/, "").trim();
  return { thinking: thinking.join("\n\n"), calls, rest };
}

function toolCallBlock(name, args) {
  let pretty = args;
  try {
    pretty = JSON.stringify(typeof args === "string" ? JSON.parse(args) : args, null, 2);
  } catch {
    /* keep raw args */
  }
  return el("div", {}, [el("span", { class: "chip" }, [name]), el("pre", { class: "msg-text" }, [pretty])]);
}

function messageCard(message, index) {
  const role = message.role ?? "?";
  const body = [];
  let content = message.content;
  let rawCalls = [];
  if (role === "assistant" && typeof content === "string" && !message.reasoning_content && !message.tool_calls) {
    const split = splitRawAssistant(content);
    if (split.thinking) body.push(el("div", { class: "msg-thinking" }, [split.thinking]));
    content = split.rest;
    rawCalls = split.calls;
  }
  if (message.reasoning_content) {
    body.push(el("div", { class: "msg-thinking" }, [message.reasoning_content]));
  }
  if (content) {
    body.push(el("pre", { class: "msg-text" }, [typeof content === "string" ? content : JSON.stringify(content, null, 2)]));
  }
  for (const raw of rawCalls) {
    let name = "tool_call";
    let args = raw;
    try {
      const parsed = JSON.parse(raw);
      name = parsed.name ?? name;
      args = parsed.arguments ?? parsed;
    } catch {
      /* unparseable call body: show verbatim */
    }
    body.push(toolCallBlock(name, args));
  }
  for (const call of message.tool_calls ?? []) {
    const fn = call.function ?? call;
    body.push(toolCallBlock(fn.name ?? "tool", fn.arguments ?? ""));
  }
  const bodyWrap = el("div", { class: `msg-body${CLOSED_BY_DEFAULT.has(role) ? " closed" : ""}` }, body);
  const header = el(
    "div",
    { class: "msg-head", onclick: () => bodyWrap.classList.toggle("closed") },
    [
      el("span", { class: `msg-role role-${role}` }, [role + (message.name ? ` · ${message.name}` : "")]),
      el("span", { class: "muted" }, [`#${index}`]),
    ],
  );
  return el("div", { class: "msg" }, [header, bodyWrap]);
}
