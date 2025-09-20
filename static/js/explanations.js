// // Function to request explanation for a movie
// async function explainRecommendation(movieId, type = 'both') {
//     try {
//         const response = await fetch('/recommender/explain/', {
//             method: 'POST',
//             headers: {
//                 'Content-Type': 'application/json',
//                 'X-CSRFToken': getCookie('csrftoken')
//             },
//             body: JSON.stringify({
//                 movie_id: movieId,
//                 type: type
//             })
//         });
        
//         const data = await response.json();
        
//         if (response.ok) {
//             displayExplanation(data);
//         } else {
//             console.error('Explanation failed:', data.error);
//             alert('Failed to generate explanation: ' + data.error);
//         }
//     } catch (error) {
//         console.error('Error:', error);
//         alert('Network error occurred');
//     }
// }

// // Function to display explanation in a modal or panel
// function displayExplanation(data) {
//     const modal = document.getElementById('explanation-modal');
//     const content = document.getElementById('explanation-content');
    
//     let html = `
//         <h3 class="text-xl font-bold mb-4">Why we recommended "${data.movie_title}"</h3>
//         <div class="space-y-4">
//     `;
    
//     if (data.explanations.shap) {
//         html += `
//             <div class="bg-blue-50 p-4 rounded">
//                 <h4 class="font-semibold text-blue-800 mb-2">SHAP Analysis</h4>
//                 <p class="text-sm text-blue-700">
//                     Prediction Score: ${data.explanations.shap.prediction?.toFixed(3)}
//                 </p>
//                 <p class="text-sm text-blue-600 mt-2">
//                     This shows how different factors contributed to your recommendation score.
//                 </p>
//             </div>
//         `;
//     }
    
//     if (data.explanations.lime) {
//         html += `
//             <div class="bg-green-50 p-4 rounded">
//                 <h4 class="font-semibold text-green-800 mb-2">LIME Analysis</h4>
//                 <p class="text-sm text-green-700">
//                     Local explanation of the most important features for this recommendation.
//                 </p>
//             </div>
//         `;
//     }
    
//     html += `
//         </div>
//         <p class="text-xs text-gray-500 mt-4">
//             Generated in ${data.generation_time_ms}ms
//         </p>
//     `;
    
//     content.innerHTML = html;
//     modal.classList.remove('hidden');
// }

// // Helper function to get CSRF token
// function getCookie(name) {
//     let cookieValue = null;
//     if (document.cookie && document.cookie !== '') {
//         const cookies = document.cookie.split(';');
//         for (let i = 0; i < cookies.length; i++) {
//             const cookie = cookies[i].trim();
//             if (cookie.substring(0, name.length + 1) === (name + '=')) {
//                 cookieValue = decodeURIComponent(cookie.substring(name.length + 1));
//                 break;
//             }
//         }
//     }
//     return cookieValue;
// }

// // Add explanation buttons to movie cards
// document.addEventListener('DOMContentLoaded', function() {
//     const movieCards = document.querySelectorAll('.movie-card');
    
//     movieCards.forEach(card => {
//         const movieId = card.dataset.movieId;
        
//         // Add "Why recommended?" button
//         const explainBtn = document.createElement('button');
//         explainBtn.className = 'explain-btn bg-blue-600 hover:bg-blue-700 text-white text-xs px-2 py-1 rounded mt-2';
//         explainBtn.textContent = 'Why recommended?';
//         explainBtn.onclick = () => explainRecommendation(movieId);
        
//         card.appendChild(explainBtn);
//     });
// });
// Attach click listeners after DOM loads
document.addEventListener('DOMContentLoaded', function() {
    document.querySelectorAll('.explain-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            const movieId = btn.closest('.movie-card').dataset.movieId;
            explainRecommendation(movieId);
        });
    });
});

// Request explanation from backend
async function explainRecommendation(movieId) {
    try {
        const response = await fetch('/recommender/explain/', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'X-CSRFToken': getCookie('csrftoken')
            },
            body: JSON.stringify({
                movie_id: movieId,
                type: "both"
            })
        });

        const data = await response.json();
        if (response.ok) {
            displayExplanation(data);
        } else {
            alert("Failed: " + (data.error || "unknown error"));
        }
    } catch (err) {
        console.error(err);
        alert("Network error");
    }
}

// Display explanation in modal
function displayExplanation(data) {
    const content = document.getElementById('explanation-content');
    let html = `
        <h3 class="text-xl font-bold mb-4">Why we recommended <span class="text-blue-600">"${data.movie_title}"</span></h3>
        <div class="space-y-4">
    `;

    if (data.explanations.shap && !data.explanations.shap.error) {
        html += `
            <div class="border-l-4 border-blue-500 pl-3">
                <h4 class="font-semibold text-blue-800">SHAP Analysis</h4>
                <p class="text-gray-700 text-sm">Prediction Score: ${data.explanations.shap.prediction?.toFixed(3)}</p>
                <pre class="text-xs text-gray-600 bg-gray-100 p-2 rounded mt-2">${JSON.stringify(data.explanations.shap.shap_values[0].slice(0,5), null, 2)} ...</pre>
            </div>
        `;
    }
    if (data.explanations.lime && !data.explanations.lime.error) {
        html += `
            <div class="border-l-4 border-green-500 pl-3">
                <h4 class="font-semibold text-green-800">LIME Analysis</h4>
                <ul class="text-sm text-gray-700 mt-1">
                    ${data.explanations.lime.lime_explanation.slice(0,5).map(([feat, val]) =>
                        `<li><b>${feat}</b>: ${val.toFixed(3)}</li>`
                    ).join("")}
                </ul>
            </div>
        `;
    }

    html += `
        </div>
        <p class="text-xs text-gray-500 mt-4">Generated in ${data.generation_time_ms}ms</p>
    `;
    content.innerHTML = html;

    document.getElementById('explanation-modal').classList.remove('hidden');
}

// Close modal
function closeExplanation() {
    document.getElementById('explanation-modal').classList.add('hidden');
}

// CSRF getter
function getCookie(name) {
    let cookieValue = null;
    if (document.cookie && document.cookie !== '') {
        const cookies = document.cookie.split(';');
        for (let i = 0; i < cookies.length; i++) {
            const cookie = cookies[i].trim();
            if (cookie.substring(0, name.length + 1) === (name + '=')) {
                cookieValue = decodeURIComponent(cookie.substring(name.length + 1));
                break;
            }
        }
    }
    return cookieValue;
}