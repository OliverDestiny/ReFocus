css = '''
.loader-container {
  display: flex; /* Use flex to align items horizontally */
  align-items: center; /* Center items vertically within the container */
  white-space: nowrap; /* Prevent line breaks within the container */
}

.loader {
  border: 8px solid #f3f3f3; /* Light grey */
  border-top: 8px solid #3498db; /* Blue */
  border-radius: 50%;
  width: 30px;
  height: 30px;
  animation: spin 2s linear infinite;
}

@keyframes spin {
  0% { transform: rotate(0deg); }
  100% { transform: rotate(360deg); }
}

/* Style the progress bar */
progress {
  appearance: none; /* Remove default styling */
  height: 20px; /* Set the height of the progress bar */
  border-radius: 5px; /* Round the corners of the progress bar */
  background-color: #f3f3f3; /* Light grey background */
  width: 100%;
}

/* Style the progress bar container */
.progress-container {
  margin-left: 20px;
  margin-right: 20px;
  flex-grow: 1; /* Allow the progress container to take up remaining space */
}

/* Set the color of the progress bar fill */
progress::-webkit-progress-value {
  background-color: #3498db; /* Blue color for the fill */
}

progress::-moz-progress-bar {
  background-color: #3498db; /* Blue color for the fill in Firefox */
}

/* Style the text on the progress bar */
progress::after {
  content: attr(value '%'); /* Display the progress value followed by '%' */
  position: absolute;
  top: 50%;
  left: 50%;
  transform: translate(-50%, -50%);
  color: white; /* Set text color */
  font-size: 14px; /* Set font size */
}

/* Style other texts */
.loader-container > span {
  margin-left: 5px; /* Add spacing between the progress bar and the text */
}

.progress-bar > .generating {
  display: none !important;
}

.progress-bar{
  height: 30px !important;
}

/* The prompt row is as tall as its content: a fixed 80px clipped the Generate column once the
   Polish Draft button became visible next to Generate, and the overflow landed on the checkbox row
   and the draft history below it. */
.type_row{
  height: auto !important;
  min-height: 80px !important;
}

/* A taller main prompt box: long prompts should not need the scrollbar. */
#positive_prompt{
  height: 140px !important;
}

.type_row_half{
  height: 32px !important;
}

.scroll-hide{
  resize: none !important;
}

.refresh_button{
  border: none !important;
  background: none !important;
  font-size: none !important;
  box-shadow: none !important;
}

/* Wide enough for "Input Image" and "Advanced" on one line; at 250px the first one wrapped. */
.advanced_check_row{
  width: 340px !important;
}

.advanced_check_row label{
  white-space: nowrap !important;
}

.min_check{
  min-width: min(1px, 100%) !important;
}

.resizable_area {
  resize: vertical;
  overflow: auto !important;
}

/* Resolution presets: the theme gives every button a min-width, which wraps four of them
   into two fat columns; zero it and shrink the text so four fit one row. */
.preset_row button {
  min-width: 0 !important;
  padding-left: 4px !important;
  padding-right: 4px !important;
}

.preset_row button span {
  font-size: 12px !important;
  white-space: nowrap !important;
}

/* Rows that must not wrap on a narrow window (Steps with its presets, Width/Height with the
   swap button). Gradio wraps the sliders of a row in an inner ".form" flex container that
   carries the theme's per-child min-widths, so both levels are unwrapped here: the outer row
   and that inner form. The sliders keep a floor so they stay draggable. */
.nowrap_row,
.nowrap_row > .form {
  flex-wrap: nowrap !important;
}

.nowrap_row > * {
  min-width: 0 !important;
}

.nowrap_row > .form > * {
  min-width: 96px !important;
}

.nowrap_row input[type="number"] {
  min-width: 58px !important;
}

.nowrap_row > button {
  min-width: 52px !important;
}

.aspect_ratios label {
    width: 140px !important;
}

.aspect_ratios label span {
    white-space: nowrap !important;
}

.aspect_ratios label input {
    margin-left: -5px !important;
}

.main_view img,
.image_gallery img {
    width: 100% !important;
    height: 100% !important;
    object-fit: contain !important;
}

.image_gallery {
    height: 100% !important;
    min-height: 600px;
}

'''
progress_html = '''
<div class="loader-container">
  <div class="loader"></div>
  <div class="progress-container">
    <progress value="*number*" max="100"></progress>
  </div>
  <span>*text*</span>
</div>
'''


def make_progress_html(number, text):
    return progress_html.replace('*number*', str(number)).replace('*text*', text)
