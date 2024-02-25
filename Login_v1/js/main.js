(function ($) {
    "use strict";

    var input = $('.validate-input .input100');

    $('.validate-form').on('submit', function (event) {
        var check = true;

        for (var i = 0; i < input.length; i++) {
            if (validate(input[i]) == false) {
                showValidate(input[i]);
                check = false;
            }
        }

        if (check) {
            // If all fields are non-empty, check for specific credentials
            var userId = $('#username').val().trim(); // Assuming the ID input has the id 'username'
            var password = $('#password').val().trim(); // Assuming the password input has the id 'password'

            if (userId !== 'admin' || password !== 'admin123') {
                // Display an alert if credentials are incorrect
                alert("Invalid credentials. Please enter 'admin' as User ID and 'admin123' as Password.");
                event.preventDefault(); // Prevent the form from submitting
            }
        }

        return check;
    });

    $('.validate-form .input100').each(function () {
        $(this).focus(function () {
            hideValidate(this);
        });
    });

    function validate(input) {
        if ($(input).attr('type') === 'text' || $(input).attr('name') === 'username') {
            // Validate for User ID (text input) - must not be empty
            if ($(input).val().trim() === '') {
                return false;
            }
        } else {
            // Default validation for other input types (e.g., email, password) - must not be empty
            if ($(input).val().trim() === '') {
                return false;
            }
        }
    }

    function showValidate(input) {
        var thisAlert = $(input).parent();
        $(thisAlert).addClass('alert-validate');
    }

    function hideValidate(input) {
        var thisAlert = $(input).parent();
        $(thisAlert).removeClass('alert-validate');
    }

})(jQuery);
